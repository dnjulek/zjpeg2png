//! Normalized 8x8 DCT/IDCT, ported statement-by-statement from jpeg2png's
//! ooura/dct.c (Takuya OOURA, http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html,
//! fft2d.zip shrtdct.c; permissive license, see notes in the C source).
//!
//! Bit-exactness: the C source declares the constants as double literals, so
//! every statement that multiplies by a constant is evaluated in f64 and
//! narrowed to f32 once at the assignment, while constant-free statements are
//! pure f32. This file replicates that mixed precision exactly; do not
//! "simplify" the casts and do not introduce @mulAdd.
//!
//! The public entry points are vectorized: the C's first loop (over columns
//! j) maps naturally to @Vector(8, f32) rows with lanes = j and identical
//! per-lane statements; the second loop (over rows) is the same pass wrapped
//! in 8x8 transposes. Per-lane math is statement-identical to the scalar
//! port, so the vector path is bit-exact (guarded by tests below).

// Cn_kR = sqrt(2.0/n) * cos(pi/2*k/n)
// Cn_kI = sqrt(2.0/n) * sin(pi/2*k/n)
// Wn_kR = cos(pi/2*k/n)
// Wn_kI = sin(pi/2*k/n)
const C8_1R: f64 = 0.49039264020161522456;
const C8_1I: f64 = 0.09754516100806413392;
const C8_2R: f64 = 0.46193976625564337806;
const C8_2I: f64 = 0.19134171618254488586;
const C8_3R: f64 = 0.41573480615127261854;
const C8_3I: f64 = 0.27778511650980111237;
const C8_4R: f64 = 0.35355339059327376220;
const W8_4R: f64 = 0.70710678118654752440;

const VF = @Vector(8, f32);
const VD = @Vector(8, f64);

inline fn wide(v: VF) VD {
    return @floatCast(v);
}

inline fn narrow(v: VD) VF {
    return @floatCast(v);
}

const vC8_1R: VD = @splat(C8_1R);
const vC8_1I: VD = @splat(C8_1I);
const vC8_2R: VD = @splat(C8_2R);
const vC8_2I: VD = @splat(C8_2I);
const vC8_3R: VD = @splat(C8_3R);
const vC8_3I: VD = @splat(C8_3I);
const vC8_4R: VD = @splat(C8_4R);
const vW8_4R: VD = @splat(W8_4R);

/// One 8-point IDCT pass over all 8 lanes at once; a[k] is the vector of the
/// algorithm's k-th elements. Statement-for-statement identical to the scalar
/// idctPass below.
inline fn idctPassVec(a: *[8]VF) void {
    var x1r: VF = narrow(vC8_1R * wide(a[1]) + vC8_1I * wide(a[7]));
    var x1i: VF = narrow(vC8_1R * wide(a[7]) - vC8_1I * wide(a[1]));
    var x3r: VF = narrow(vC8_3R * wide(a[3]) + vC8_3I * wide(a[5]));
    var x3i: VF = narrow(vC8_3R * wide(a[5]) - vC8_3I * wide(a[3]));
    var xr: VF = x1r - x3r;
    var xi: VF = x1i + x3i;
    x1r += x3r;
    x3i -= x1i;
    x1i = narrow(vW8_4R * wide(xr + xi));
    x3r = narrow(vW8_4R * wide(xr - xi));
    xr = narrow(vC8_2R * wide(a[2]) + vC8_2I * wide(a[6]));
    xi = narrow(vC8_2R * wide(a[6]) - vC8_2I * wide(a[2]));
    var x0r: VF = narrow(vC8_4R * wide(a[0] + a[4]));
    var x0i: VF = narrow(vC8_4R * wide(a[0] - a[4]));
    const x2r: VF = x0r - xr;
    const x2i: VF = x0i - xi;
    x0r += xr;
    x0i += xi;
    a[0] = x0r + x1r;
    a[7] = x0r - x1r;
    a[2] = x0i + x1i;
    a[5] = x0i - x1i;
    a[4] = x2r - x3i;
    a[3] = x2r + x3i;
    a[6] = x2i - x3r;
    a[1] = x2i + x3r;
}

/// One 8-point DCT pass over all 8 lanes at once.
inline fn dctPassVec(a: *[8]VF) void {
    const x0r: VF = a[0] + a[7];
    var x1r: VF = a[0] - a[7];
    const x0i: VF = a[2] + a[5];
    var x1i: VF = a[2] - a[5];
    const x2r: VF = a[4] + a[3];
    var x3r: VF = a[4] - a[3];
    const x2i: VF = a[6] + a[1];
    var x3i: VF = a[6] - a[1];
    var xr: VF = x0r + x2r;
    var xi: VF = x0i + x2i;
    a[0] = narrow(vC8_4R * wide(xr + xi));
    a[4] = narrow(vC8_4R * wide(xr - xi));
    xr = x0r - x2r;
    xi = x0i - x2i;
    a[2] = narrow(vC8_2R * wide(xr) - vC8_2I * wide(xi));
    a[6] = narrow(vC8_2R * wide(xi) + vC8_2I * wide(xr));
    xr = narrow(vW8_4R * wide(x1i - x3i));
    x1i = narrow(vW8_4R * wide(x1i + x3i));
    x3i = x1i - x3r;
    x1i += x3r;
    x3r = x1r - xr;
    x1r += xr;
    a[1] = narrow(vC8_1R * wide(x1r) - vC8_1I * wide(x1i));
    a[7] = narrow(vC8_1R * wide(x1i) + vC8_1I * wide(x1r));
    a[3] = narrow(vC8_3R * wide(x3r) - vC8_3I * wide(x3i));
    a[5] = narrow(vC8_3R * wide(x3i) + vC8_3I * wide(x3r));
}

/// In-place 8x8 transpose of eight 8-lane rows, three butterfly stages.
inline fn transpose8(m: *[8]VF) void {
    // stage 1: interleave element pairs of adjacent rows
    const s0 = @shuffle(f32, m[0], m[1], [8]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
    const s1 = @shuffle(f32, m[0], m[1], [8]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
    const s2 = @shuffle(f32, m[2], m[3], [8]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
    const s3 = @shuffle(f32, m[2], m[3], [8]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
    const s4 = @shuffle(f32, m[4], m[5], [8]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
    const s5 = @shuffle(f32, m[4], m[5], [8]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
    const s6 = @shuffle(f32, m[6], m[7], [8]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
    const s7 = @shuffle(f32, m[6], m[7], [8]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
    // stage 2: interleave 2-lane chunks at distance 2
    const t0 = @shuffle(f32, s0, s2, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const t1 = @shuffle(f32, s0, s2, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const t2 = @shuffle(f32, s1, s3, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const t3 = @shuffle(f32, s1, s3, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const t4 = @shuffle(f32, s4, s6, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const t5 = @shuffle(f32, s4, s6, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const t6 = @shuffle(f32, s5, s7, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const t7 = @shuffle(f32, s5, s7, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    // stage 3: interleave 4-lane chunks at distance 4
    m[0] = @shuffle(f32, t0, t4, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
    m[1] = @shuffle(f32, t0, t4, [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 });
    m[2] = @shuffle(f32, t1, t5, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
    m[3] = @shuffle(f32, t1, t5, [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 });
    m[4] = @shuffle(f32, t2, t6, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
    m[5] = @shuffle(f32, t2, t6, [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 });
    m[6] = @shuffle(f32, t3, t7, [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 });
    m[7] = @shuffle(f32, t3, t7, [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 });
}

/// Normalized 8x8 IDCT, in place. Natural (raster) coefficient order.
pub fn idct8x8s(a: *[64]f32) void {
    var rows: [8]VF = undefined;
    inline for (0..8) |r| rows[r] = a[r * 8 ..][0..8].*;
    idctPassVec(&rows); // C's first loop: lanes = column index j
    transpose8(&rows);
    idctPassVec(&rows); // C's second loop, via transpose
    transpose8(&rows);
    inline for (0..8) |r| a[r * 8 ..][0..8].* = rows[r];
}

/// Normalized 8x8 DCT, in place. Natural (raster) coefficient order.
pub fn dct8x8s(a: *[64]f32) void {
    var rows: [8]VF = undefined;
    inline for (0..8) |r| rows[r] = a[r * 8 ..][0..8].*;
    dctPassVec(&rows);
    transpose8(&rows);
    dctPassVec(&rows);
    transpose8(&rows);
    inline for (0..8) |r| a[r * 8 ..][0..8].* = rows[r];
}

// ---------------------------------------------------------------------------
// Scalar reference implementation (the original statement-level port), kept
// for the bit-equality cross-check below.

inline fn idctPass(a: *[64]f32, comptime stride: usize, base: usize) void {
    var x1r: f32 = @floatCast(C8_1R * @as(f64, a[base + 1 * stride]) + C8_1I * @as(f64, a[base + 7 * stride]));
    var x1i: f32 = @floatCast(C8_1R * @as(f64, a[base + 7 * stride]) - C8_1I * @as(f64, a[base + 1 * stride]));
    var x3r: f32 = @floatCast(C8_3R * @as(f64, a[base + 3 * stride]) + C8_3I * @as(f64, a[base + 5 * stride]));
    var x3i: f32 = @floatCast(C8_3R * @as(f64, a[base + 5 * stride]) - C8_3I * @as(f64, a[base + 3 * stride]));
    var xr: f32 = x1r - x3r;
    var xi: f32 = x1i + x3i;
    x1r += x3r;
    x3i -= x1i;
    x1i = @floatCast(W8_4R * @as(f64, xr + xi));
    x3r = @floatCast(W8_4R * @as(f64, xr - xi));
    xr = @floatCast(C8_2R * @as(f64, a[base + 2 * stride]) + C8_2I * @as(f64, a[base + 6 * stride]));
    xi = @floatCast(C8_2R * @as(f64, a[base + 6 * stride]) - C8_2I * @as(f64, a[base + 2 * stride]));
    var x0r: f32 = @floatCast(C8_4R * @as(f64, a[base + 0 * stride] + a[base + 4 * stride]));
    var x0i: f32 = @floatCast(C8_4R * @as(f64, a[base + 0 * stride] - a[base + 4 * stride]));
    const x2r: f32 = x0r - xr;
    const x2i: f32 = x0i - xi;
    x0r += xr;
    x0i += xi;
    a[base + 0 * stride] = x0r + x1r;
    a[base + 7 * stride] = x0r - x1r;
    a[base + 2 * stride] = x0i + x1i;
    a[base + 5 * stride] = x0i - x1i;
    a[base + 4 * stride] = x2r - x3i;
    a[base + 3 * stride] = x2r + x3i;
    a[base + 6 * stride] = x2i - x3r;
    a[base + 1 * stride] = x2i + x3r;
}

inline fn dctPass(a: *[64]f32, comptime stride: usize, base: usize) void {
    const x0r: f32 = a[base + 0 * stride] + a[base + 7 * stride];
    var x1r: f32 = a[base + 0 * stride] - a[base + 7 * stride];
    const x0i: f32 = a[base + 2 * stride] + a[base + 5 * stride];
    var x1i: f32 = a[base + 2 * stride] - a[base + 5 * stride];
    const x2r: f32 = a[base + 4 * stride] + a[base + 3 * stride];
    var x3r: f32 = a[base + 4 * stride] - a[base + 3 * stride];
    const x2i: f32 = a[base + 6 * stride] + a[base + 1 * stride];
    var x3i: f32 = a[base + 6 * stride] - a[base + 1 * stride];
    var xr: f32 = x0r + x2r;
    var xi: f32 = x0i + x2i;
    a[base + 0 * stride] = @floatCast(C8_4R * @as(f64, xr + xi));
    a[base + 4 * stride] = @floatCast(C8_4R * @as(f64, xr - xi));
    xr = x0r - x2r;
    xi = x0i - x2i;
    a[base + 2 * stride] = @floatCast(C8_2R * @as(f64, xr) - C8_2I * @as(f64, xi));
    a[base + 6 * stride] = @floatCast(C8_2R * @as(f64, xi) + C8_2I * @as(f64, xr));
    xr = @floatCast(W8_4R * @as(f64, x1i - x3i));
    x1i = @floatCast(W8_4R * @as(f64, x1i + x3i));
    x3i = x1i - x3r;
    x1i += x3r;
    x3r = x1r - xr;
    x1r += xr;
    a[base + 1 * stride] = @floatCast(C8_1R * @as(f64, x1r) - C8_1I * @as(f64, x1i));
    a[base + 7 * stride] = @floatCast(C8_1R * @as(f64, x1i) + C8_1I * @as(f64, x1r));
    a[base + 3 * stride] = @floatCast(C8_3R * @as(f64, x3r) - C8_3I * @as(f64, x3i));
    a[base + 5 * stride] = @floatCast(C8_3R * @as(f64, x3i) + C8_3I * @as(f64, x3r));
}

fn idct8x8sScalar(a: *[64]f32) void {
    for (0..8) |j| idctPass(a, 8, j);
    for (0..8) |j| idctPass(a, 1, j * 8);
}

fn dct8x8sScalar(a: *[64]f32) void {
    for (0..8) |j| dctPass(a, 8, j);
    for (0..8) |j| dctPass(a, 1, j * 8);
}

// ---------------------------------------------------------------------------

const std = @import("std");

test "transpose8 is correct and involutive" {
    var m: [8]VF = undefined;
    for (0..8) |r| {
        var row: [8]f32 = undefined;
        for (0..8) |c| row[c] = @floatFromInt(r * 8 + c);
        m[r] = row;
    }
    var t = m;
    transpose8(&t);
    for (0..8) |r| {
        const t_row: [8]f32 = t[r];
        for (0..8) |c| {
            const m_row: [8]f32 = m[c];
            try std.testing.expectEqual(m_row[r], t_row[c]);
        }
    }
    transpose8(&t);
    for (0..8) |r| {
        try std.testing.expectEqual(m[r], t[r]);
    }
}

test "vector dct/idct bit-identical to scalar reference" {
    var prng = std.Random.DefaultPrng.init(0xdc7);
    const random = prng.random();
    for (0..200) |_| {
        var block: [64]f32 = undefined;
        for (&block) |*v| v.* = (random.float(f32) - 0.5) * 4096.0;
        var vec_d = block;
        var sca_d = block;
        dct8x8s(&vec_d);
        dct8x8sScalar(&sca_d);
        try std.testing.expectEqualSlices(u32, &@as([64]u32, @bitCast(sca_d)), &@as([64]u32, @bitCast(vec_d)));
        var vec_i = block;
        var sca_i = block;
        idct8x8s(&vec_i);
        idct8x8sScalar(&sca_i);
        try std.testing.expectEqualSlices(u32, &@as([64]u32, @bitCast(sca_i)), &@as([64]u32, @bitCast(vec_i)));
    }
}

test "dct/idct roundtrip" {
    var prng = std.Random.DefaultPrng.init(0x6a7065677a);
    const random = prng.random();
    var block: [64]f32 = undefined;
    var orig: [64]f32 = undefined;
    for (&block, &orig) |*b, *o| {
        const v = (random.float(f32) - 0.5) * 2048.0;
        b.* = v;
        o.* = v;
    }
    dct8x8s(&block);
    idct8x8s(&block);
    for (block, orig) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 0.005);
    }
}

test "idct of pure DC" {
    var block: [64]f32 = @splat(0);
    block[0] = 1024.0;
    idct8x8s(&block);
    for (block) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 128.0), v, 0.001);
    }
}

test "idct bit-exact vs C reference" {
    // Block 10261 of the original 1920x1080 test image's luma channel
    // (dequantized coefficients) and the float bit patterns the C jpeg2png
    // (gcc -O3 -msse2 -mfpmath=sse -ffp-contract=off) produced for it. Guards
    // the f64/f32 mixed-precision discipline of the port. The capture is baked
    // in below, so this test needs no image file.
    // zig fmt: off
    const input = [64]i32{
        -176, -320,   94,   55,    0,   0, 0, 0,
        -240, -200, -220,  240,  -67,   0, 0, 0,
         -94,  495,   65, -134,    0,   0, 0, 0,
        -220,  -55,  130,    0,  -72,   0, 0, 0,
           0,   65,  -67,    0,   80,   0, 0, 0,
          65,   67,   72,  -80,   87,   0, 0, 0,
         -65,  -67,    0,    0,    0,   0, 0, 0,
          67,    0,    0,    0,    0,   0, 0, 0,
    };
    const expected_bits = [64]u32{
        0xC29508E1, 0xC2C36DCB, 0xC2F7E9E3, 0xC2FB826E, 0xC2C9B630, 0xC2A9DD43, 0xC2D5934B, 0xC30D80A0,
        0xC30446D6, 0xC2D7159F, 0xC2E8615C, 0xC2E01DFF, 0xC16A5C18, 0x42A2B16E, 0x41986C58, 0xC2F7E214,
        0xC3028CDA, 0xC30CAC31, 0xC2E490B2, 0xC1BAEFEC, 0x42B383BB, 0x43115BD3, 0x42E735A1, 0x42759652,
        0xC2BCD386, 0xC3181B01, 0xC2F5A696, 0x413634CC, 0x42D4655C, 0x42D05A44, 0x42C1414E, 0x42F62C26,
        0xC30D7D6D, 0xC3168797, 0xC30F9F96, 0xC2A3B750, 0x42047292, 0x4300F1D2, 0x430D5481, 0x42D3F192,
        0xC2F04271, 0xC2CA8DC4, 0xC2F9D2DF, 0xC3265D3F, 0xC2FE802A, 0xC0A7A250, 0x42BEBADA, 0x42FCE829,
        0x42FAEECD, 0x42DE3AC0, 0x4294DB5E, 0xC0C360F8, 0xC2D4A420, 0xC2F74942, 0xBF035580, 0x4313F030,
        0x43078C4A, 0x42F449EB, 0x42E98F36, 0x42939AE2, 0xC23D3C41, 0xC3112CD9, 0xC2BCB477, 0x41D4C07C,
    };
    // zig fmt: on
    var block: [64]f32 = undefined;
    for (&block, input) |*b, v| b.* = @floatFromInt(v);
    idct8x8s(&block);
    for (block, expected_bits) |got, want| {
        try std.testing.expectEqual(want, @as(u32, @bitCast(got)));
    }
}
