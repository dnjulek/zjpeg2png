//! YCbCr -> RGB rendering and PNG writing — port of jpeg2png's png.c, with
//! libpng replaced by zigimg's PNG encoder. The pixel VALUES are the spec
//! (bit-exact vs C); the PNG container bytes are not (different deflate).
const std = @import("std");
const zigimg = @import("zigimg");

const Coef = @import("jpeg.zig").Coef;

// clamp to RGB range; C's CLAMP(x, 0., 255.) test order
fn clamp(x: f32) f32 {
    return if (x > 255.0) 255.0 else if (x < 0.0) 0.0 else x;
}

inline fn p(data: []const f32, x: usize, y: usize, w: usize) f32 {
    return data[y * w + x];
}

/// Render to 8-bit RGB bytes (len w*h*3), computing exactly what png.c's
/// write loop computes: the YCbCr->RGB inner expressions are evaluated in
/// f64 (double literals promote the floats), narrowed to f32 once at clamp's
/// parameter, multiplied by the f32 bitfactor and truncated.
/// Each channel is indexed with its own dimensions (they can differ from the
/// image's and from each other's, e.g. in separate-components mode).
/// Null chroma renders channel 0 as gray (R=G=B), like the let-def fork's
/// `cb ? ... : 0.0` for 1- and 4-component files.
pub fn renderRgb8(gpa: std.mem.Allocator, w: u32, h: u32, y: *const Coef, cb: ?*const Coef, cr: ?*const Coef) error{OutOfMemory}![]u8 {
    const out = try gpa.alloc(u8, @as(usize, w) * h * 3);
    const bitfactor: f32 = 1.0; // C: (1 << 8) / 256.
    for (0..h) |i| {
        for (0..w) |j| {
            const yi = p(y.fdata, j, i, y.w);
            const cbi: f32 = if (cb) |coef| p(coef.fdata, j, i, coef.w) else 0.0;
            const cri: f32 = if (cr) |coef| p(coef.fdata, j, i, coef.w) else 0.0;

            // YCbCr -> RGB
            const r: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) + 1.402 * @as(f64, cri))) * bitfactor);
            const g: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) - 0.34414 * @as(f64, cbi) - 0.71414 * @as(f64, cri))) * bitfactor);
            const b: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) + 1.772 * @as(f64, cbi))) * bitfactor);

            const here = out[(@as(usize, i) * w + j) * 3 ..][0..3];
            here[0] = @truncate(r);
            here[1] = @truncate(g);
            here[2] = @truncate(b);
        }
    }
    return out;
}

/// 16-bit variant (--16-bits-png): same math with bitfactor 256; the u16
/// values correspond to the big-endian byte pairs png.c writes.
pub fn renderRgb16(gpa: std.mem.Allocator, w: u32, h: u32, y: *const Coef, cb: ?*const Coef, cr: ?*const Coef) error{OutOfMemory}![]u16 {
    const out = try gpa.alloc(u16, @as(usize, w) * h * 3);
    const bitfactor: f32 = 256.0; // C: (1 << 16) / 256.
    for (0..h) |i| {
        for (0..w) |j| {
            const yi = p(y.fdata, j, i, y.w);
            const cbi: f32 = if (cb) |coef| p(coef.fdata, j, i, coef.w) else 0.0;
            const cri: f32 = if (cr) |coef| p(coef.fdata, j, i, coef.w) else 0.0;

            const r: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) + 1.402 * @as(f64, cri))) * bitfactor);
            const g: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) - 0.34414 * @as(f64, cbi) - 0.71414 * @as(f64, cri))) * bitfactor);
            const b: u32 = @intFromFloat(clamp(@floatCast(@as(f64, yi) + 1.772 * @as(f64, cbi))) * bitfactor);

            const here = out[(@as(usize, i) * w + j) * 3 ..][0..3];
            here[0] = @intCast(r & 0xFFFF);
            here[1] = @intCast(g & 0xFFFF);
            here[2] = @intCast(b & 0xFFFF);
        }
    }
    return out;
}

/// Write rendered 8-bit RGB samples to a PNG file.
pub fn writeRgb8Png(gpa: std.mem.Allocator, io: std.Io, path: []const u8, w: u32, h: u32, rgb: []const u8) !void {
    var write_buffer: [4096]u8 = undefined;
    const image = try zigimg.Image.fromRawPixelsOwned(w, h, rgb, .rgb24);
    try image.writeToFilePath(gpa, io, path, &write_buffer, .{ .png = .{} });
}

/// Write rendered 16-bit RGB samples to a PNG file.
pub fn writeRgb16Png(gpa: std.mem.Allocator, io: std.Io, path: []const u8, w: u32, h: u32, rgb: []const u16) !void {
    var write_buffer: [4096]u8 = undefined;
    const image = try zigimg.Image.fromRawPixelsOwned(w, h, std.mem.sliceAsBytes(rgb), .rgb48);
    try image.writeToFilePath(gpa, io, path, &write_buffer, .{ .png = .{} });
}
