//! Adapted from zigimg (https://github.com/zigimg/zigimg), MIT license.
//! Source: src/formats/jpeg/quantization.zig — t-81 section 2.4.1.
//! Tables are de-zigzagged during parsing, so they are stored in natural
//! (raster) order — same convention as libjpeg's quantval.
const std = @import("std");

const Image = @import("zigimg").Image;

const ZigzagOffsets = @import("utils.zig").ZigzagOffsets;

pub const Table = union(enum) {
    q8: [64]u8,
    q16: [64]u16,

    pub fn read(precision: u8, reader: *std.Io.Reader) Image.ReadError!Table {
        // 0 = 8 bits, 1 = 16 bits
        switch (precision) {
            0 => {
                var table = Table{ .q8 = undefined };

                var offset: usize = 0;
                while (offset < 64) : (offset += 1) {
                    const value = try reader.takeByte();
                    table.q8[ZigzagOffsets[offset]] = value;
                }

                return table;
            },
            1 => {
                var table = Table{ .q16 = undefined };

                var offset: usize = 0;
                while (offset < 64) : (offset += 1) {
                    const value = try reader.takeInt(u16, .big);
                    table.q16[ZigzagOffsets[offset]] = value;
                }

                return table;
            },
            else => return Image.ReadError.InvalidData,
        }
    }
};
