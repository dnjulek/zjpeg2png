//! The decode_file pipeline of jpeg2png.c, minus the file I/O:
//! read coefficients -> decode + unbox -> compute (joint or per-channel)
//! -> luma fixup -> render RGB.
const std = @import("std");

const jpeg_mod = @import("jpeg.zig");
const boxing = @import("box.zig");
const compute_mod = @import("compute.zig");
const png = @import("png.zig");
const Logger = @import("logger.zig").Logger;
const Progress = @import("progressbar.zig").Progress;

pub const Options = struct {
    iterations: [4]u32 = .{ 50, 50, 50, 50 },
    weights: [4]f32 = .{ 0.3, 0, 0, 0 },
    pweights: [4]f32 = .{ 0.001, 0.001, 0.001, 0.001 },
    png_bits: u8 = 8,
    /// optimize all components jointly (the default); false = -s
    all_together: bool = true,
    /// in separate-components mode, how many channels to optimize in
    /// parallel (1 = serial); pixel results are identical either way
    channel_threads: u32 = 1,
    /// threads used INSIDE one compute() call (1 = the bit-exact serial
    /// reference path; >1 is deterministic but may differ from the serial
    /// result by last-ulp rounding at row-strip seams)
    compute_threads: u32 = 1,
};

pub const Rendered = struct {
    w: u32,
    h: u32,
    /// w*h*3 bytes when png_bits == 8
    rgb8: []u8 = &.{},
    /// w*h*3 samples when png_bits == 16 (values = big-endian byte pairs in C)
    rgb16: []u16 = &.{},

    pub fn deinit(self: *Rendered, gpa: std.mem.Allocator) void {
        if (self.rgb8.len != 0) gpa.free(self.rgb8);
        if (self.rgb16.len != 0) gpa.free(self.rgb16);
        self.* = undefined;
    }
};

/// Run the whole pipeline on an in-memory JPEG, returning the rendered RGB
/// samples (the exact values write_png would store in the PNG).
pub fn decodeToPixels(
    gpa: std.mem.Allocator,
    jpeg_bytes: []const u8,
    opts: Options,
    pb: ?Progress,
    log: *const Logger,
) !Rendered {
    var jpeg = try jpeg_mod.readCoefficients(gpa, jpeg_bytes);
    defer jpeg.deinit(gpa);

    const ncomp = jpeg.ncomp;

    // decode jpg normally
    for (jpeg.coefs[0..ncomp]) |*coef| {
        try jpeg_mod.decodeCoefficients(gpa, coef);
    }
    for (jpeg.coefs[0..ncomp]) |*coef| {
        const temp = try gpa.alloc(f32, @as(usize, coef.w) * coef.h);
        boxing.unbox(coef.fdata, temp, coef.w, coef.h);
        gpa.free(coef.fdata);
        coef.fdata = temp;
    }

    // smooth; joint optimization only applies to 3-component (YCbCr) files,
    // as in the let-def fork — gray and CMYK always go channel by channel
    if (opts.all_together and ncomp == 3) {
        var channel_log = log.*;
        channel_log.channel = 3;
        try compute_mod.compute(gpa, jpeg.coefs[0..3], &channel_log, pb, opts.weights[0], opts.pweights[0..], opts.iterations[0], opts.compute_threads);
    } else if (opts.channel_threads > 1) {
        // C parallelizes this loop with OpenMP; channels are independent
        const inner_threads: u32 = @max(1, opts.compute_threads / @as(u32, @intCast(ncomp)));
        var channel_logs: [4]Logger = undefined;
        var channel_errors: [4]?error{OutOfMemory} = .{ null, null, null, null };
        var threads: [4]?std.Thread = .{ null, null, null, null };
        for (0..ncomp) |c| {
            channel_logs[c] = log.*;
            channel_logs[c].channel = @intCast(c);
            threads[c] = std.Thread.spawn(.{}, channelWorker, .{
                gpa, &jpeg.coefs, c, &channel_logs[c], pb, &opts, inner_threads, &channel_errors[c],
            }) catch null;
            if (threads[c] == null) {
                channelWorker(gpa, &jpeg.coefs, c, &channel_logs[c], pb, &opts, inner_threads, &channel_errors[c]);
            }
        }
        for (0..ncomp) |c| {
            if (threads[c]) |thread| thread.join();
        }
        for (0..ncomp) |c| {
            if (channel_errors[c]) |err| return err;
        }
    } else {
        for (0..ncomp) |c| {
            var channel_log = log.*;
            channel_log.channel = @intCast(c);
            try compute_mod.compute(gpa, jpeg.coefs[c .. c + 1], &channel_log, pb, opts.weights[c], opts.pweights[c .. c + 1], opts.iterations[c], opts.compute_threads);
        }
    }

    // fixup luma range
    for (jpeg.coefs[0].fdata) |*v| {
        v.* += 128.0;
    }

    // render; non-3-component files render channel 0 as gray (NULL chroma in
    // the C fork)
    var rendered = Rendered{ .w = jpeg.w, .h = jpeg.h };
    errdefer rendered.deinit(gpa);
    const y = &jpeg.coefs[0];
    const cb: ?*const jpeg_mod.Coef = if (ncomp == 3) &jpeg.coefs[1] else null;
    const cr: ?*const jpeg_mod.Coef = if (ncomp == 3) &jpeg.coefs[2] else null;
    if (opts.png_bits == 8) {
        rendered.rgb8 = try png.renderRgb8(gpa, jpeg.w, jpeg.h, y, cb, cr);
    } else {
        rendered.rgb16 = try png.renderRgb16(gpa, jpeg.w, jpeg.h, y, cb, cr);
    }
    return rendered;
}

fn channelWorker(
    gpa: std.mem.Allocator,
    coefs: *[4]jpeg_mod.Coef,
    c: usize,
    log: *Logger,
    pb: ?Progress,
    opts: *const Options,
    inner_threads: u32,
    err_out: *?error{OutOfMemory},
) void {
    // channels already run in parallel here; inner_threads splits the budget
    compute_mod.compute(gpa, coefs[c .. c + 1], log, pb, opts.weights[c], opts.pweights[c .. c + 1], opts.iterations[c], inner_threads) catch |err| {
        err_out.* = err;
    };
}

/// Full file-to-file pipeline (decode_file in C).
pub fn decodeFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    infile: []const u8,
    outfile: []const u8,
    opts: Options,
    pb: ?Progress,
    log: *const Logger,
) !void {
    const jpeg_bytes = std.Io.Dir.cwd().readFileAlloc(io, infile, gpa, .unlimited) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.CouldNotOpenInput,
    };
    defer gpa.free(jpeg_bytes);

    var rendered = try decodeToPixels(gpa, jpeg_bytes, opts, pb, log);
    defer rendered.deinit(gpa);

    const write_result = if (opts.png_bits == 8)
        png.writeRgb8Png(gpa, io, outfile, rendered.w, rendered.h, rendered.rgb8)
    else
        png.writeRgb16Png(gpa, io, outfile, rendered.w, rendered.h, rendered.rgb16);
    write_result catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.CouldNotWriteOutput,
    };
}
