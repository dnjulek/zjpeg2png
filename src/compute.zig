//! The optimizer — a function-by-function port of jpeg2png's compute.c:
//! subgradient descent with projection and FISTA acceleration, minimizing
//! TV + weight*TGV2 + pweight*(DCT coefficient deviation)^2 over the box of
//! images that quantize back to the source JPEG.
//!
//! The hot kernels are vectorized with @Vector(VLEN, f32), following the
//! design of the C project's own compute_simd_step.c, which is documented
//! (and was verified) to produce results bit-identical to the scalar code.
//! The per-lane math is statement-identical to the scalar functions (kept
//! below as the border/tail implementations), and the gradient scatter order
//! preserves the scalar accumulation order on overlapping cells — see the
//! "N.B." comments. The vector path is therefore bit-exact, with one
//! documented exception: the tv/tv2/prob_dist objective *accumulators* (CSV
//! log only, never pixels) may differ in the last digits.
//!
//! Bit-exactness rules (do not "clean up"):
//! - everything is f32 except the spots where C promotes to double; those are
//!   written as explicit f64 expressions narrowed once with @floatCast;
//! - float expressions keep the exact shape/order of the C source; no
//!   @mulAdd, no reassociation;
//! - computeNorm's serial f64 accumulator feeds the step size and is
//!   load-bearing — keep it serial.
const std = @import("std");

const Coef = @import("jpeg.zig").Coef;
const boxing = @import("box.zig");
const dct = @import("dct.zig");
const Logger = @import("logger.zig").Logger;
const Progress = @import("progressbar.zig").Progress;

/// Vector width for the hot loops; capped at 8 because image widths are only
/// guaranteed to be multiples of 8.
pub const VLEN: comptime_int = @min(8, std.simd.suggestVectorLength(f32) orelse 4);
const VF = @Vector(VLEN, f32);

/// shared zero-length target for emptied aligned slices (never written/freed)
var empty_aligned: [0]f32 align(64) = .{};

const AlignedSlice = []align(64) f32;

inline fn loadV(data: []const f32, i: usize) VF {
    return data[i..][0..VLEN].*;
}

inline fn storeV(data: []f32, i: usize, v: VF) void {
    data[i..][0..VLEN].* = v;
}

inline fn rmwAddV(data: []f32, i: usize, v: VF) void {
    storeV(data, i, loadV(data, i) + v);
}

/// Run work(ctx, i) for i in 0..n on up to `threads` threads (the calling
/// thread is one of them). Plain spawn/join per call; the per-iteration cost
/// is microseconds against multi-millisecond phases.
fn runParallel(comptime Ctx: type, comptime work: fn (Ctx, usize) void, ctx: Ctx, n: usize, threads: u32) void {
    if (threads <= 1 or n <= 1) {
        for (0..n) |i| work(ctx, i);
        return;
    }
    const Pool = struct {
        fn worker(c: Ctx, next: *std.atomic.Value(usize), total: usize) void {
            while (true) {
                const i = next.fetchAdd(1, .monotonic);
                if (i >= total) return;
                work(c, i);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    var handles: [15]?std.Thread = @splat(null);
    const nthreads: usize = @min(threads, n);
    for (handles[0 .. @min(nthreads, 16) - 1]) |*h| {
        h.* = std.Thread.spawn(.{}, Pool.worker, .{ ctx, &next, n }) catch null;
    }
    Pool.worker(ctx, &next, n);
    for (handles) |h| {
        if (h) |t| t.join();
    }
}

/// Strip partitioning for row-parallel TV/TGV: strips are processed in two
/// passes (even strips, then odd strips), so concurrently running strips
/// never write the same rows (scatter reaches at most one row beyond the
/// strip). The result is deterministic for a given strip height, but the
/// accumulation order at strip seams differs from the serial code — tiny
/// (last-ulp) differences relative to the bit-exact serial path.
const max_strips = 64;

fn stripHeight(rows: usize, threads: u32) usize {
    var sh = (rows + @as(usize, threads) * 2 - 1) / (@as(usize, threads) * 2);
    if (sh < 2) sh = 2;
    const min_sh = (rows + max_strips - 1) / max_strips;
    return @max(sh, min_sh);
}

/// Working buffers for each component, mirroring C's struct aux.
/// 64-byte aligned for cache-line-clean vector access.
const Aux = struct {
    /// DCT coefficients of the current (projected) image, for stepProb
    cos: AlignedSlice,
    /// gradient (derivative) of the objective function
    obj_gradient: AlignedSlice,
    /// temp[0] = pixel differences in x direction
    /// temp[1] = pixel differences in y direction
    /// also used differently in computeProjection
    temp: [2]AlignedSlice,
    /// image data
    fdata: AlignedSlice,
    /// previous step image data for FISTA
    fista: AlignedSlice,
};

inline fn sqf(x: f32) f32 {
    return x * x;
}

/// index image, planar layout
inline fn p(data: []f32, x: usize, y: usize, w: usize) *f32 {
    return &data[y * w + x];
}

// compute objective gradient for the distance of DCT coefficients from
// normal decoding. N.B. destroys cos
fn computeStepProb(w: usize, h: usize, alpha: f32, coef: *const Coef, cos: []f32, obj_gradient: []f32) f64 {
    _ = h;
    var prob_dist: f64 = 0.0;
    const block_w = coef.w / 8;
    const block_h = coef.h / 8;
    const valpha: VF = @splat(alpha);
    const full_res = coef.w_samp == 1 and coef.h_samp == 1;
    for (0..block_h) |block_y| {
        for (0..block_w) |block_x| {
            const i = block_y * block_w + block_x;
            const cosb = cos[i * 64 ..][0..64];
            var j: usize = 0;
            while (j < 64) : (j += VLEN) {
                const data_f: VF = @floatFromInt(@as(@Vector(VLEN, i16), coef.data[i * 64 + j ..][0..VLEN].*));
                const quant_f: VF = @floatFromInt(@as(@Vector(VLEN, u16), coef.quant_table[j..][0..VLEN].*));
                var cv = loadV(cosb, j);
                cv -= data_f * quant_f;
                const t = cv / quant_f;
                const dist = t * t;
                const dist_arr: [VLEN]f32 = dist;
                for (dist_arr) |lane| prob_dist += @as(f64, lane); // objective function (log-only)
                cv = cv / (quant_f * quant_f); // derivative
                storeV(cosb, j, cv);
            }
            dct.idct8x8s(cosb);
            // unbox and possibly upsample derivative
            if (full_res) {
                for (0..8) |in_y| {
                    const y = block_y * 8 + in_y;
                    const x = block_x * 8;
                    var jj: usize = 0;
                    while (jj < 8) : (jj += VLEN) {
                        rmwAddV(obj_gradient, y * w + x + jj, valpha * loadV(cosb, in_y * 8 + jj));
                    }
                }
            } else {
                for (0..8) |in_y| {
                    for (0..8) |in_x| {
                        const j2 = in_y * 8 + in_x;
                        const cx = block_x * 8 + in_x;
                        const cy = block_y * 8 + in_y;
                        for (0..coef.h_samp) |sy| {
                            for (0..coef.w_samp) |sx| {
                                const y = cy * coef.h_samp + sy;
                                const x = cx * coef.w_samp + sx;
                                p(obj_gradient, x, y, w).* += alpha * cosb[j2];
                            }
                        }
                    }
                }
            }
        }
    }
    return @as(f64, alpha) * (0.5 * prob_dist);
}

// compute objective gradient for TV for one pixel (scalar; used for borders)
fn computeStepTvInner(w: usize, h: usize, nchannel: usize, auxs: []Aux, alpha: f32, x: usize, y: usize, tv: *f64) void {
    var g_xs: [4]f32 = @splat(0);
    var g_ys: [4]f32 = @splat(0);
    for (0..nchannel) |c| {
        const aux = &auxs[c];
        // forward difference x
        g_xs[c] = if (x >= w - 1) 0.0 else p(aux.fdata, x + 1, y, w).* - p(aux.fdata, x, y, w).*;
        // forward difference y
        g_ys[c] = if (y >= h - 1) 0.0 else p(aux.fdata, x, y + 1, w).* - p(aux.fdata, x, y, w).*;
    }
    // norm
    var g_norm: f32 = 0.0;
    for (0..nchannel) |c| {
        g_norm += sqf(g_xs[c]);
        g_norm += sqf(g_ys[c]);
    }
    g_norm = @sqrt(g_norm);
    tv.* += @as(f64, alpha * g_norm); // objective function
    // compute derivatives (see notes)
    for (0..nchannel) |c| {
        const g_x = g_xs[c];
        const g_y = g_ys[c];
        const aux = &auxs[c];
        if (g_norm != 0) {
            p(aux.obj_gradient, x, y, w).* += alpha * -(g_x + g_y) / g_norm;
            if (x < w - 1) {
                p(aux.obj_gradient, x + 1, y, w).* += alpha * g_x / g_norm;
            }
            if (y < h - 1) {
                p(aux.obj_gradient, x, y + 1, w).* += alpha * g_y / g_norm;
            }
        }
    }
    // store for use in tv2
    for (0..nchannel) |c| {
        const aux = &auxs[c];
        p(aux.temp[0], x, y, w).* = g_xs[c];
        p(aux.temp[1], x, y, w).* = g_ys[c];
    }
}

// vector TV inner: VLEN pixels at once; valid only for x+VLEN < w and
// y < h-1 (all neighbor loads in range). Per-lane math identical to the
// scalar inner.
inline fn computeStepTvInnerVec(w: usize, nchannel: usize, auxs: []Aux, alpha: f32, x: usize, y: usize, tv: *f64) void {
    var g_xs: [4]VF = undefined;
    var g_ys: [4]VF = undefined;
    for (0..nchannel) |c| {
        const aux = &auxs[c];
        const here = loadV(aux.fdata, y * w + x);
        // forward differences
        g_xs[c] = loadV(aux.fdata, y * w + x + 1) - here;
        g_ys[c] = loadV(aux.fdata, (y + 1) * w + x) - here;
    }
    var g_norm: VF = @splat(0);
    for (0..nchannel) |c| {
        g_norm += g_xs[c] * g_xs[c];
        g_norm += g_ys[c] * g_ys[c];
    }
    g_norm = @sqrt(g_norm);

    // lanes are consecutive x positions: same accumulation order as scalar
    const norm_arr: [VLEN]f32 = g_norm;
    for (norm_arr) |lane| tv.* += @as(f64, alpha * lane);

    // zeroes -> infinity so the gradient terms become 0 instead of NaN
    // (the C simd cmpeq+or trick, via @select)
    const div = @select(f32, g_norm == @as(VF, @splat(0)), @as(VF, @splat(std.math.inf(f32))), g_norm);
    const malpha: VF = @splat(alpha);
    for (0..nchannel) |c| {
        const g_x = g_xs[c];
        const g_y = g_ys[c];
        const aux = &auxs[c];
        // N.B. for the same exact result as the scalar version, the
        // objective gradient at x+1 must be computed before x
        rmwAddV(aux.obj_gradient, y * w + x + 1, malpha * g_x / div);
        rmwAddV(aux.obj_gradient, y * w + x, malpha * (-(g_x + g_y)) / div);
        rmwAddV(aux.obj_gradient, (y + 1) * w + x, malpha * g_y / div);
    }
    // store for use in tv2
    for (0..nchannel) |c| {
        storeV(auxs[c].temp[0], y * w + x, g_xs[c]);
        storeV(auxs[c].temp[1], y * w + x, g_ys[c]);
    }
}

// one full row of TV (vector body + scalar tail); valid for y < h-1
inline fn tvRow(w: usize, h: usize, nchannel: usize, auxs: []Aux, alpha: f32, y: usize, tv: *f64) void {
    var x: usize = 0;
    while (x + VLEN < w) : (x += VLEN) {
        computeStepTvInnerVec(w, nchannel, auxs, alpha, x, y, tv);
    }
    while (x < w) : (x += 1) {
        computeStepTvInner(w, h, nchannel, auxs, alpha, x, y, tv);
    }
}

const TvStripCtx = struct {
    w: usize,
    h: usize,
    nchannel: usize,
    auxs: []Aux,
    alpha: f32,
    strip_h: usize,
    rows: usize, // rows 0..rows are striped (exclusive)
    parity: usize,
    partials: *[max_strips]f64,
};

fn tvStripWork(ctx: TvStripCtx, k: usize) void {
    const strip = 2 * k + ctx.parity;
    const start = strip * ctx.strip_h;
    if (start >= ctx.rows) return;
    const end = @min(start + ctx.strip_h, ctx.rows);
    var tv: f64 = 0.0;
    for (start..end) |y| {
        tvRow(ctx.w, ctx.h, ctx.nchannel, ctx.auxs, ctx.alpha, y, &tv);
    }
    ctx.partials[strip] = tv;
}

// compute objective gradient for TV
fn computeStepTv(w: usize, h: usize, nchannel: usize, auxs: []Aux, threads: u32) f64 {
    var tv: f64 = 0.0;
    std.debug.assert(nchannel <= 4);
    // C computes this per pixel inside the inner function:
    // float alpha = 1./sqrtf(nchannel) — a double division narrowed to float.
    const alpha: f32 = @floatCast(1.0 / @as(f64, @sqrt(@as(f32, @floatFromInt(nchannel)))));
    if (w < 2 * VLEN or w % VLEN != 0) {
        for (0..h) |y| {
            for (0..w) |x| {
                computeStepTvInner(w, h, nchannel, auxs, alpha, x, y, &tv);
            }
        }
        return tv;
    }
    const rows = h - 1;
    if (threads > 1 and rows >= 8) {
        // row-striped, even/odd passes (deterministic; seam accumulation
        // order differs from serial — see strip notes above)
        var partials: [max_strips]f64 = @splat(0.0);
        const strip_h = stripHeight(rows, threads);
        const nstrips = (rows + strip_h - 1) / strip_h;
        const ctx = TvStripCtx{
            .w = w,
            .h = h,
            .nchannel = nchannel,
            .auxs = auxs,
            .alpha = alpha,
            .strip_h = strip_h,
            .rows = rows,
            .parity = 0,
            .partials = &partials,
        };
        runParallel(TvStripCtx, tvStripWork, ctx, (nstrips + 1) / 2, threads);
        var odd_ctx = ctx;
        odd_ctx.parity = 1;
        runParallel(TvStripCtx, tvStripWork, odd_ctx, nstrips / 2, threads);
        for (partials[0..nstrips]) |partial| tv += partial;
    } else {
        for (0..rows) |y| {
            tvRow(w, h, nchannel, auxs, alpha, y, &tv);
        }
    }
    for (0..w) |x| {
        computeStepTvInner(w, h, nchannel, auxs, alpha, x, h - 1, &tv);
    }
    return tv;
}

// compute objective gradient for second order TGV for one pixel (scalar)
fn computeStepTv2Inner(w: usize, h: usize, nchannel: usize, auxs: []Aux, alpha: f32, x: usize, y: usize, tv2: *f64) void {
    var g_xxs: [4]f32 = @splat(0);
    var g_xy_syms: [4]f32 = @splat(0);
    var g_yys: [4]f32 = @splat(0);

    for (0..nchannel) |c| {
        const aux = &auxs[c];

        // backward difference x
        g_xxs[c] = if (x <= 0) 0.0 else p(aux.temp[0], x, y, w).* - p(aux.temp[0], x - 1, y, w).*;
        // backward difference x
        const g_yx: f32 = if (x <= 0) 0.0 else p(aux.temp[1], x, y, w).* - p(aux.temp[1], x - 1, y, w).*;
        // backward difference y
        const g_xy: f32 = if (y <= 0) 0.0 else p(aux.temp[0], x, y, w).* - p(aux.temp[0], x, y - 1, w).*;
        // backward difference y
        g_yys[c] = if (y <= 0) 0.0 else p(aux.temp[1], x, y, w).* - p(aux.temp[1], x, y - 1, w).*;
        // symmetrize; C: (g_xy + g_yx) / 2. — f32 add, f64 divide, narrowed
        g_xy_syms[c] = @floatCast(@as(f64, g_xy + g_yx) / 2.0);
    }
    // norm
    var g2_norm: f32 = 0.0;
    for (0..nchannel) |c| {
        g2_norm += sqf(g_xxs[c]) + 2 * sqf(g_xy_syms[c]) + sqf(g_yys[c]);
    }
    g2_norm = @sqrt(g2_norm);

    tv2.* += @as(f64, alpha * g2_norm); // objective function

    // compute derivatives (see notes)
    if (g2_norm != 0.0) {
        for (0..nchannel) |c| {
            const g_xx = g_xxs[c];
            const g_yy = g_yys[c];
            const g_xy_sym = g_xy_syms[c];
            const aux = &auxs[c];

            p(aux.obj_gradient, x, y, w).* += alpha * (-(2 * g_xx + 2 * g_xy_sym + 2 * g_yy) / g2_norm);
            if (x > 0) {
                p(aux.obj_gradient, x - 1, y, w).* += alpha * ((g_xy_sym + g_xx) / g2_norm);
            }
            if (x < w - 1) {
                p(aux.obj_gradient, x + 1, y, w).* += alpha * ((g_xy_sym + g_xx) / g2_norm);
            }
            if (y > 0) {
                p(aux.obj_gradient, x, y - 1, w).* += alpha * ((g_yy + g_xy_sym) / g2_norm);
            }
            if (y < h - 1) {
                p(aux.obj_gradient, x, y + 1, w).* += alpha * ((g_yy + g_xy_sym) / g2_norm);
            }
            if (x < w - 1 and y > 0) {
                p(aux.obj_gradient, x + 1, y - 1, w).* += alpha * ((-g_xy_sym) / g2_norm);
            }
            if (x > 0 and y < h - 1) {
                p(aux.obj_gradient, x - 1, y + 1, w).* += alpha * ((-g_xy_sym) / g2_norm);
            }
        }
    }
}

// vector TGV inner: VLEN pixels at once; valid for VLEN <= x < w-VLEN and
// 0 < y < h-1. Scatter windows in the C simd right-to-left order, which
// preserves the scalar accumulation order on overlapping cells.
inline fn computeStepTv2InnerVec(w: usize, nchannel: usize, auxs: []Aux, alpha: f32, x: usize, y: usize, tv2: *f64) void {
    var g_xxs: [4]VF = undefined;
    var g_xy_syms: [4]VF = undefined;
    var g_yys: [4]VF = undefined;
    const vtwo: VF = @splat(2.0);

    for (0..nchannel) |c| {
        const aux = &auxs[c];
        const g_x = loadV(aux.temp[0], y * w + x);
        const g_y = loadV(aux.temp[1], y * w + x);
        // backward differences
        g_xxs[c] = g_x - loadV(aux.temp[0], y * w + x - 1);
        const g_yx = g_y - loadV(aux.temp[1], y * w + x - 1);
        const g_xy = g_x - loadV(aux.temp[0], (y - 1) * w + x);
        g_yys[c] = g_y - loadV(aux.temp[1], (y - 1) * w + x);
        // symmetrize
        g_xy_syms[c] = (g_xy + g_yx) / vtwo;
    }
    var g2_norm: VF = @splat(0);
    for (0..nchannel) |c| {
        g2_norm += g_xxs[c] * g_xxs[c] + vtwo * (g_xy_syms[c] * g_xy_syms[c]) + g_yys[c] * g_yys[c];
    }
    g2_norm = @sqrt(g2_norm);

    const norm_arr: [VLEN]f32 = g2_norm;
    for (norm_arr) |lane| tv2.* += @as(f64, alpha * lane);

    const div = @select(f32, g2_norm == @as(VF, @splat(0)), @as(VF, @splat(std.math.inf(f32))), g2_norm);
    const malpha: VF = @splat(alpha);
    for (0..nchannel) |c| {
        const g_xx = g_xxs[c];
        const g_yy = g_yys[c];
        const g_xy_sym = g_xy_syms[c];
        const aux = &auxs[c];

        // N.B. for the same exact result as the scalar version, the
        // objective gradient is computed from right to left
        rmwAddV(aux.obj_gradient, (y - 1) * w + x + 1, malpha * ((-g_xy_sym) / div));
        rmwAddV(aux.obj_gradient, y * w + x + 1, malpha * ((g_xy_sym + g_xx) / div));
        rmwAddV(aux.obj_gradient, (y - 1) * w + x, malpha * ((g_yy + g_xy_sym) / div));
        rmwAddV(aux.obj_gradient, y * w + x, malpha * (-(vtwo * g_xx + vtwo * g_xy_sym + vtwo * g_yy) / div));
        rmwAddV(aux.obj_gradient, (y + 1) * w + x, malpha * ((g_yy + g_xy_sym) / div));
        rmwAddV(aux.obj_gradient, y * w + x - 1, malpha * ((g_xy_sym + g_xx) / div));
        rmwAddV(aux.obj_gradient, (y + 1) * w + x - 1, malpha * ((-g_xy_sym) / div));
    }
}

// one full row of TGV (scalar borders + vector body); valid for 0 < y < h-1
inline fn tv2Row(w: usize, h: usize, nchannel: usize, auxs: []Aux, alpha2: f32, y: usize, tv2: *f64) void {
    for (0..VLEN) |x| {
        computeStepTv2Inner(w, h, nchannel, auxs, alpha2, x, y, tv2);
    }
    var x: usize = VLEN;
    while (x + VLEN < w) : (x += VLEN) {
        computeStepTv2InnerVec(w, nchannel, auxs, alpha2, x, y, tv2);
    }
    while (x < w) : (x += 1) {
        computeStepTv2Inner(w, h, nchannel, auxs, alpha2, x, y, tv2);
    }
}

const Tv2StripCtx = struct {
    w: usize,
    h: usize,
    nchannel: usize,
    auxs: []Aux,
    alpha2: f32,
    strip_h: usize,
    rows: usize, // strips cover rows 1 .. 1+rows
    parity: usize,
    partials: *[max_strips]f64,
};

fn tv2StripWork(ctx: Tv2StripCtx, k: usize) void {
    const strip = 2 * k + ctx.parity;
    const start = strip * ctx.strip_h;
    if (start >= ctx.rows) return;
    const end = @min(start + ctx.strip_h, ctx.rows);
    var tv2: f64 = 0.0;
    for (1 + start..1 + end) |y| {
        tv2Row(ctx.w, ctx.h, ctx.nchannel, ctx.auxs, ctx.alpha2, y, &tv2);
    }
    ctx.partials[strip] = tv2;
}

// compute objective gradient for second order TGV
fn computeStepTv2(w: usize, h: usize, nchannel: usize, auxs: []Aux, alpha: f32, threads: u32) f64 {
    var tv2: f64 = 0.0;
    // C computes this per pixel inside the inner function:
    // alpha = alpha * 1./sqrtf(nchannel) — (alpha * 1.0) / sqrt in double,
    // narrowed to float.
    const alpha2: f32 = @floatCast(@as(f64, alpha) * 1.0 / @as(f64, @sqrt(@as(f32, @floatFromInt(nchannel)))));
    if (w < 2 * VLEN or w % VLEN != 0 or h < 2) {
        for (0..h) |y| {
            for (0..w) |x| {
                computeStepTv2Inner(w, h, nchannel, auxs, alpha2, x, y, &tv2);
            }
        }
        return tv2;
    }
    for (0..w) |x| {
        computeStepTv2Inner(w, h, nchannel, auxs, alpha2, x, 0, &tv2);
    }
    const rows = h - 2; // rows 1..h-1
    if (threads > 1 and rows >= 8) {
        var partials: [max_strips]f64 = @splat(0.0);
        const strip_h = stripHeight(rows, threads);
        const nstrips = (rows + strip_h - 1) / strip_h;
        const ctx = Tv2StripCtx{
            .w = w,
            .h = h,
            .nchannel = nchannel,
            .auxs = auxs,
            .alpha2 = alpha2,
            .strip_h = strip_h,
            .rows = rows,
            .parity = 0,
            .partials = &partials,
        };
        runParallel(Tv2StripCtx, tv2StripWork, ctx, (nstrips + 1) / 2, threads);
        var odd_ctx = ctx;
        odd_ctx.parity = 1;
        runParallel(Tv2StripCtx, tv2StripWork, odd_ctx, nstrips / 2, threads);
        for (partials[0..nstrips]) |partial| tv2 += partial;
    } else {
        for (1..h - 1) |y| {
            tv2Row(w, h, nchannel, auxs, alpha2, y, &tv2);
        }
    }
    for (0..w) |x| {
        computeStepTv2Inner(w, h, nchannel, auxs, alpha2, x, h - 1, &tv2);
    }
    return tv2;
}

// compute Euclidean norm; C accumulates f32 squares into a double and takes
// sqrtf of the narrowed sum. Serial on purpose: the sum order feeds the step
// size and is part of the bit-exact contract.
fn computeNorm(data: []const f32) f32 {
    var norm: f64 = 0.0;
    for (data) |d| {
        norm += @as(f64, sqf(d));
    }
    return @sqrt(@as(f32, @floatCast(norm)));
}

// make step in the direction of the objective gradient with distance step_size
fn computeDoStep(fdata: []f32, obj_gradient: []const f32, step_size: f32) void {
    const norm = computeNorm(obj_gradient);
    if (norm != 0.0) {
        const vstep: VF = @splat(step_size);
        const vnorm: VF = @splat(norm);
        var i: usize = 0;
        while (i + VLEN <= fdata.len) : (i += VLEN) {
            storeV(fdata, i, loadV(fdata, i) - vstep * (loadV(obj_gradient, i) / vnorm));
        }
        while (i < fdata.len) : (i += 1) {
            fdata[i] = fdata[i] - step_size * (obj_gradient[i] / norm);
        }
    }
}

const ProbPhaseCtx = struct {
    w: usize,
    h: usize,
    coefs: []Coef,
    auxs: []Aux,
    pweights: []const f32,
    prob_dists: *[4]f64,
};

fn probPhaseWork(ctx: ProbPhaseCtx, c: usize) void {
    const aux = &ctx.auxs[c];
    // initialize gradient
    @memset(aux.obj_gradient, 0.0);
    // DCT coefficient distance
    if (ctx.pweights[c] != 0.0) {
        const p_alpha: f32 = ctx.pweights[c] * 2 * 255 * @sqrt(@as(f32, 2.0));
        ctx.prob_dists[c] = computeStepProb(ctx.w, ctx.h, p_alpha, &ctx.coefs[c], aux.cos, aux.obj_gradient);
    }
}

const DoStepCtx = struct {
    auxs: []Aux,
    step_size: f32,
};

fn doStepWork(ctx: DoStepCtx, c: usize) void {
    computeDoStep(ctx.auxs[c].fdata, ctx.auxs[c].obj_gradient, ctx.step_size);
}

// compute objective gradient and make step
fn computeStep(
    w: usize,
    h: usize,
    nchannel: usize,
    coefs: []Coef,
    auxs: []Aux,
    step_size: f32,
    weight: f32,
    pweights: []const f32,
    log: *Logger,
    threads: u32,
) f64 {
    var total_alpha: f32 = 0.0;

    // gradient init + DCT coefficient distance, channel-parallel (independent
    // buffers; bit-exact). The scalar reductions happen in channel order.
    var prob_dists: [4]f64 = @splat(0.0);
    runParallel(ProbPhaseCtx, probPhaseWork, .{
        .w = w,
        .h = h,
        .coefs = coefs,
        .auxs = auxs,
        .pweights = pweights,
        .prob_dists = &prob_dists,
    }, nchannel, threads);
    var prob_dist: f64 = 0.0;
    for (0..nchannel) |c| {
        if (pweights[c] != 0.0) {
            total_alpha += pweights[c] * 2 * 255 * @sqrt(@as(f32, 2.0));
            prob_dist += prob_dists[c];
        }
    }

    // TV
    total_alpha += @floatFromInt(nchannel);
    const tv = computeStepTv(w, h, nchannel, auxs, threads);

    // TGV second order
    var tv2: f64 = 0.0;
    if (weight != 0.0) {
        // C: weight / sqrtf(4 / 2) — note the integer division 4/2
        const alpha: f32 = weight / @sqrt(@as(f32, 2.0));
        total_alpha += alpha * @as(f32, @floatFromInt(nchannel));
        tv2 = computeStepTv2(w, h, nchannel, auxs, alpha, threads);
    }

    // do step, channel-parallel (bit-exact)
    runParallel(DoStepCtx, doStepWork, .{ .auxs = auxs, .step_size = step_size }, nchannel, threads);

    // log objective values
    const objective = (tv + tv2 + prob_dist) / @as(f64, total_alpha);
    log.log(objective, prob_dist, tv, tv2);

    return objective;
}

// initialize working buffers
fn auxInit(gpa: std.mem.Allocator, w: usize, h: usize, coef: *Coef, aux: *Aux) error{OutOfMemory}!void {
    aux.* = .{
        .cos = empty_aligned[0..],
        .obj_gradient = empty_aligned[0..],
        .temp = .{ empty_aligned[0..], empty_aligned[0..] },
        .fdata = empty_aligned[0..],
        .fista = empty_aligned[0..],
    };

    aux.cos = try gpa.alignedAlloc(f32, .@"64", @as(usize, coef.h) * coef.w);
    const blocks = @as(usize, coef.h / 8) * (coef.w / 8);
    for (0..blocks) |i| {
        for (0..64) |j| {
            // int multiply then a single int->float rounding (as in C)
            aux.cos[i * 64 + j] = @floatFromInt(@as(i32, coef.data[i * 64 + j]) * coef.quant_table[j]);
        }
    }

    for (0..2) |i| {
        aux.temp[i] = try gpa.alignedAlloc(f32, .@"64", w * h);
    }
    aux.obj_gradient = try gpa.alignedAlloc(f32, .@"64", w * h);

    // nearest upsample to the full optimization resolution, clamped at edges
    const fdata = try gpa.alignedAlloc(f32, .@"64", w * h);
    for (0..h) |y| {
        for (0..w) |x| {
            const cy = @min(y / coef.h_samp, coef.h - 1);
            const cx = @min(x / coef.w_samp, coef.w - 1);
            p(fdata, x, y, w).* = p(coef.fdata, cx, cy, coef.w).*;
        }
    }
    aux.fdata = fdata;
    gpa.free(coef.fdata);
    coef.fdata = &.{};

    const fista = try gpa.alignedAlloc(f32, .@"64", w * h);
    @memcpy(fista, fdata);
    aux.fista = fista;
}

// destroy working buffers, except the fdata that is returned; idempotent
// (the error path may visit an already-destroyed aux)
fn auxDestroy(gpa: std.mem.Allocator, aux: *Aux) void {
    if (aux.cos.len != 0) gpa.free(aux.cos);
    for (0..2) |i| {
        if (aux.temp[i].len != 0) gpa.free(aux.temp[i]);
    }
    if (aux.obj_gradient.len != 0) gpa.free(aux.obj_gradient);
    if (aux.fista.len != 0) gpa.free(aux.fista);
    aux.* = .{
        .cos = empty_aligned[0..],
        .obj_gradient = empty_aligned[0..],
        .temp = .{ empty_aligned[0..], empty_aligned[0..] },
        .fdata = empty_aligned[0..],
        .fista = empty_aligned[0..],
    };
}

// clamp the DCT values to the interval that quantizes to our jpg
fn clampDct(coef: *const Coef, boxed: []f32, blocks: usize) void {
    const vhalf: VF = @splat(0.5);
    for (0..blocks) |i| {
        var j: usize = 0;
        while (j < 64) : (j += VLEN) {
            const data_f: VF = @floatFromInt(@as(@Vector(VLEN, i16), coef.data[i * 64 + j ..][0..VLEN].*));
            const quant_f: VF = @floatFromInt(@as(@Vector(VLEN, u16), coef.quant_table[j..][0..VLEN].*));
            const lo = (data_f - vhalf) * quant_f;
            const hi = (data_f + vhalf) * quant_f;
            const v = loadV(boxed, i * 64 + j);
            // C's order: max(min, min(max, data)) — NaN-free input
            storeV(boxed, i * 64 + j, @max(lo, @min(hi, v)));
        }
    }
}

// compute projection of data onto the feasible set defined by our jpg
fn computeProjection(w: usize, h: usize, aux: *Aux, coef: *const Coef) void {
    const blocks = @as(usize, coef.h / 8) * (coef.w / 8);
    const boxed = aux.temp[0];
    const resample = !(coef.w == w and coef.h == h);
    const subsampled = if (resample) aux.temp[1] else aux.fdata;

    // downsample and keep the difference
    // more formally, decompose each subsampling block in the direction of our
    // subsampling vector (a vector of ones)
    if (resample) {
        for (0..coef.h) |cy| {
            for (0..coef.w) |cx| {
                var mean: f32 = 0.0;
                for (0..coef.h_samp) |sy| {
                    for (0..coef.w_samp) |sx| {
                        const y = cy * coef.h_samp + sy;
                        const x = cx * coef.w_samp + sx;
                        mean += p(aux.fdata, x, y, w).*;
                    }
                }
                mean /= @as(f32, @floatFromInt(coef.w_samp * coef.h_samp));
                p(subsampled, cx, cy, coef.w).* = mean;
                for (0..coef.h_samp) |sy| {
                    for (0..coef.w_samp) |sx| {
                        const y = cy * coef.h_samp + sy;
                        const x = cx * coef.w_samp + sx;
                        p(aux.fdata, x, y, w).* -= mean;
                    }
                }
            }
        }
    }

    // project onto our DCT box
    boxing.box(subsampled, boxed, coef.w, coef.h);

    for (0..blocks) |i| {
        dct.dct8x8s(boxed[i * 64 ..][0..64]);
    }

    clampDct(coef, boxed, blocks);

    // save a copy of the DCT values for stepProb
    @memcpy(aux.cos[0 .. @as(usize, coef.w) * coef.h], boxed[0 .. @as(usize, coef.w) * coef.h]);

    for (0..blocks) |i| {
        dct.idct8x8s(boxed[i * 64 ..][0..64]);
    }

    boxing.unbox(boxed, subsampled, coef.w, coef.h);

    // add back the difference (orthogonal to our subsampling vector)
    if (resample) {
        for (0..coef.h) |cy| {
            for (0..coef.w) |cx| {
                const mean = p(subsampled, cx, cy, coef.w).*;
                for (0..coef.h_samp) |sy| {
                    for (0..coef.w_samp) |sx| {
                        const y = cy * coef.h_samp + sy;
                        const x = cx * coef.w_samp + sx;
                        p(aux.fdata, x, y, w).* += mean;
                    }
                }
            }
        }
    }
}

const FistaCtx = struct {
    auxs: []Aux,
    factor: f32,
};

fn fistaWork(ctx: FistaCtx, c: usize) void {
    const aux = &ctx.auxs[c];
    const vfactor: VF = @splat(ctx.factor);
    const n = aux.fista.len;
    var j: usize = 0;
    while (j + VLEN <= n) : (j += VLEN) {
        const fd = loadV(aux.fdata, j);
        storeV(aux.fista, j, fd + vfactor * (fd - loadV(aux.fista, j)));
    }
    while (j < n) : (j += 1) {
        aux.fista[j] = aux.fdata[j] + ctx.factor * (aux.fdata[j] - aux.fista[j]);
    }
    std.mem.swap(AlignedSlice, &aux.fdata, &aux.fista);
}

const ProjectionCtx = struct {
    w: usize,
    h: usize,
    auxs: []Aux,
    coefs: []Coef,
};

fn projectionWork(ctx: ProjectionCtx, c: usize) void {
    computeProjection(ctx.w, ctx.h, &ctx.auxs[c], &ctx.coefs[c]);
}

/// subgradient method with iteration steps; on return each coef's fdata holds
/// the optimized image at full optimization resolution (coef.w/coef.h are
/// updated to match, exactly like the C code).
///
/// threads > 1 parallelizes inside the computation: the per-channel phases
/// (prob/do_step/projection/FISTA) are bit-exact under threading; the TV/TGV
/// row strips are deterministic but may differ from the serial result by
/// last-ulp rounding at strip seams. threads == 1 is the bit-exact reference
/// path.
pub fn compute(
    gpa: std.mem.Allocator,
    coefs: []Coef,
    log: *Logger,
    progress: ?Progress,
    weight: f32,
    pweights: []const f32,
    iterations: u32,
    threads: u32,
) error{OutOfMemory}!void {
    const nchannel = coefs.len;
    std.debug.assert(nchannel >= 1 and nchannel <= 4);
    std.debug.assert(pweights.len >= nchannel);

    var w: usize = 0;
    var h: usize = 0;
    for (coefs) |*coef| {
        w = @max(w, @as(usize, coef.w) * coef.w_samp);
        h = @max(h, @as(usize, coef.h) * coef.h_samp);
    }
    std.debug.assert(w % 8 == 0);
    std.debug.assert(h % 8 == 0);

    // working buffers per channel
    var auxs_storage: [4]Aux = undefined;
    const auxs = auxs_storage[0..nchannel];
    var initialized: usize = 0;
    errdefer for (auxs[0..@min(initialized + 1, nchannel)]) |*aux| {
        if (aux.fdata.len != 0) gpa.free(aux.fdata);
        auxDestroy(gpa, aux);
    };
    for (0..nchannel) |c| {
        try auxInit(gpa, w, h, &coefs[c], &auxs[c]);
        initialized += 1;
    }

    const radius: f32 = @sqrt(@as(f32, @floatFromInt(h)) * @as(f32, @floatFromInt(w))) / 2; // radius of [-0.5, 0.5]^(h*w)
    const step_size: f32 = radius / @sqrt(@as(f32, @floatFromInt(1 + iterations)));
    var t: f32 = 1;
    for (0..iterations) |i| {
        log.iteration = @intCast(i);

        // FISTA, channel-parallel (bit-exact)
        const tnext: f32 = (1 + @sqrt(1 + 4 * sqf(t))) / 2;
        const factor: f32 = (t - 1) / tnext;
        runParallel(FistaCtx, fistaWork, .{ .auxs = auxs, .factor = factor }, nchannel, threads);
        t = tnext;

        // take a step
        _ = computeStep(w, h, nchannel, coefs, auxs, step_size, weight, pweights, log, threads);
        // project back onto feasible set, channel-parallel (bit-exact)
        runParallel(ProjectionCtx, projectionWork, .{ .w = w, .h = h, .auxs = auxs, .coefs = coefs }, nchannel, threads);
        if (progress) |prog| {
            prog.inc();
        }
    }

    // return result; copied out so the aligned working buffer stays internal
    for (0..nchannel) |c| {
        const aux = &auxs[c];
        const coef = &coefs[c];
        const result = try gpa.alloc(f32, aux.fdata.len);
        @memcpy(result, aux.fdata);
        coef.fdata = result;
        gpa.free(aux.fdata);
        aux.fdata = empty_aligned[0..];
        coef.w = @intCast(w);
        coef.h = @intCast(h);
        auxDestroy(gpa, aux);
    }
}
