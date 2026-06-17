//! Conversion between 8x8-block layout (consecutive 64-float blocks, blocks
//! ordered row-major) and planar scanline layout. Port of jpeg2png's box.c.
const std = @import("std");

/// Convert from 8x8 blocks to normal (planar) order.
pub fn unbox(in: []const f32, out: []f32, w: u32, h: u32) void {
    std.debug.assert(w % 8 == 0);
    std.debug.assert(h % 8 == 0);
    std.debug.assert(in.len >= @as(usize, w) * h and out.len >= @as(usize, w) * h);
    var i: usize = 0;
    for (0..h / 8) |block_y| {
        for (0..w / 8) |block_x| {
            for (0..8) |in_y| {
                for (0..8) |in_x| {
                    out[(block_y * 8 + in_y) * w + (block_x * 8 + in_x)] = in[i];
                    i += 1;
                }
            }
        }
    }
}

/// Convert from normal (planar) order to 8x8 blocks.
pub fn box(in: []const f32, out: []f32, w: u32, h: u32) void {
    std.debug.assert(w % 8 == 0);
    std.debug.assert(h % 8 == 0);
    std.debug.assert(in.len >= @as(usize, w) * h and out.len >= @as(usize, w) * h);
    var i: usize = 0;
    for (0..h / 8) |block_y| {
        for (0..w / 8) |block_x| {
            for (0..8) |in_y| {
                for (0..8) |in_x| {
                    out[i] = in[(block_y * 8 + in_y) * w + (block_x * 8 + in_x)];
                    i += 1;
                }
            }
        }
    }
}

test "box/unbox roundtrip" {
    const w = 24;
    const h = 16;
    var planar: [w * h]f32 = undefined;
    for (&planar, 0..) |*v, i| v.* = @floatFromInt(i);
    var blocked: [w * h]f32 = undefined;
    var back: [w * h]f32 = undefined;
    box(&planar, &blocked, w, h);
    unbox(&blocked, &back, w, h);
    try std.testing.expectEqualSlices(f32, &planar, &back);
    // spot-check block layout: first block holds the top-left 8x8 tile
    try std.testing.expectEqual(@as(f32, 0), blocked[0]);
    try std.testing.expectEqual(@as(f32, 7), blocked[7]);
    try std.testing.expectEqual(@as(f32, w), blocked[8]);
    try std.testing.expectEqual(@as(f32, 8), blocked[64]);
}
