//! `zjpeg2png info` subcommand: print JPEG metadata (dimensions, coding,
//! subsampling, per-channel block geometry, quantization tables) using the
//! coefficient reader — no optimization is run.
const std = @import("std");
const zli = @import("zli");
const zjpeg2png = @import("zjpeg2png");

const die = @import("root.zig").die;

pub fn register(init_options: zli.InitOptions) !*zli.Command {
    const cmd = try zli.Command.init(init_options, .{
        .name = "info",
        .description = "Inspect a JPEG file (dimensions, subsampling, quantization tables)",
    }, run);

    try cmd.addPositionalArg(.{
        .name = "file",
        .description = "JPEG file to inspect",
        .required = true,
    });

    return cmd;
}

/// conventional J:a:b name for the common subsampling layouts, derived from
/// the chroma subsampling factors (w_samp, h_samp)
fn subsamplingName(w_samp: u32, h_samp: u32) ?[]const u8 {
    if (w_samp == 1 and h_samp == 1) return "4:4:4 (no chroma subsampling)";
    if (w_samp == 2 and h_samp == 1) return "4:2:2";
    if (w_samp == 2 and h_samp == 2) return "4:2:0";
    if (w_samp == 1 and h_samp == 2) return "4:4:0";
    if (w_samp == 4 and h_samp == 1) return "4:1:1";
    return null;
}

fn run(ctx: zli.CommandContext) !void {
    const gpa = ctx.allocator;
    const io = ctx.io;
    const writer = ctx.writer;
    const path = ctx.getArg("file").?;

    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30)) catch
        die("could not open input file `{s}`", .{path});
    defer gpa.free(bytes);

    var jpeg = zjpeg2png.jpeg.readCoefficients(gpa, bytes) catch |err| switch (err) {
        error.OutOfMemory => die("could not allocate memory", .{}),
        error.InvalidComponentCount => die("only jpegs with 1 to 4 components are supported (gray, rgb/yuv or cmyk)", .{}),
        error.InvalidQuantTable => die("invalid quantization table", .{}),
        error.InvalidCoefSize => die("jpeg invalid coef size", .{}),
        error.ImageTooBig => die("jpeg is too big to fit in memory", .{}),
        else => die("could not decode jpeg `{s}`: {t}", .{ path, err }),
    };
    defer jpeg.deinit(gpa);

    const color_model: []const u8 = switch (jpeg.ncomp) {
        1 => "grayscale",
        3 => "YCbCr",
        4 => "CMYK/YCCK",
        else => "unknown",
    };
    try writer.print("{s} ({B})\n", .{ path, bytes.len });
    try writer.print("  dimensions:  {d} x {d}\n", .{ jpeg.w, jpeg.h });
    try writer.print("  coding:      {s}, 8-bit Huffman, {d} component{s} ({s})\n", .{
        if (jpeg.progressive) "progressive" else "baseline",
        jpeg.ncomp,
        if (jpeg.ncomp == 1) "" else "s",
        color_model,
    });
    if (jpeg.ncomp == 3) {
        const cb = &jpeg.coefs[1];
        if (subsamplingName(cb.w_samp, cb.h_samp)) |name| {
            try writer.print("  subsampling: {s}\n", .{name});
        } else {
            try writer.print("  subsampling: {d}x{d} chroma\n", .{ cb.w_samp, cb.h_samp });
        }
    }
    try writer.print("\n", .{});

    const channel_names: [4][]const u8 = switch (jpeg.ncomp) {
        3 => .{ "Y", "Cb", "Cr", "" },
        else => .{ "c0", "c1", "c2", "c3" },
    };
    for (jpeg.coefs[0..jpeg.ncomp], channel_names[0..jpeg.ncomp]) |*coef, name| {
        try writer.print("  {s: <2} {d: >5} x {d: <5} ({d} x {d} blocks, sampling {d}x{d})\n", .{
            name, coef.w, coef.h, coef.w / 8, coef.h / 8, coef.w_samp, coef.h_samp,
        });
        try writer.print("     quantization table:\n", .{});
        for (0..8) |row| {
            try writer.print("      ", .{});
            for (0..8) |col| {
                try writer.print(" {d: >4}", .{coef.quant_table[row * 8 + col]});
            }
            try writer.print("\n", .{});
        }
    }
    try writer.flush();
}
