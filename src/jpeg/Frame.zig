//! Adapted from zigimg (https://github.com/zigimg/zigimg), MIT license.
//! Source: src/formats/jpeg/Frame.zig — trimmed to entropy-decode state only:
//! the dequantization/IDCT/render stages were removed because jpeg2png needs
//! the raw quantized coefficients.
//!
//! All components share one MCU-padded block grid: block_storage[slot][c] is
//! the 8x8 block (natural order, i32) of component c at grid slot
//! row * block_width_actual + col, where a component's block (by, bx) lives at
//! row = (by/v_c)*v_max + by%v_c, col = (bx/h_c)*h_max + bx%h_c.
const std = @import("std");

const Image = @import("zigimg").Image;

const Markers = @import("utils.zig").Markers;
const FrameHeader = @import("FrameHeader.zig");
const QuantizationTable = @import("quantization.zig").Table;
const HuffmanTable = @import("huffman.zig").Table;

const MAX_COMPONENTS = @import("utils.zig").MAX_COMPONENTS;
const Block = @import("utils.zig").Block;

const Frame = @This();

allocator: std.mem.Allocator,
frame_header: FrameHeader,
quantization_tables: *[4]?QuantizationTable,
dc_huffman_tables: *[4]?HuffmanTable,
ac_huffman_tables: *[4]?HuffmanTable,
block_storage: [][MAX_COMPONENTS]Block,
frame_type: Markers = undefined,

block_height: u32 = 0,
block_width: u32 = 0,
block_width_actual: u32 = 0,
block_height_actual: u32 = 0,

horizontal_sampling_factor_max: usize = 0,
vertical_sampling_factor_max: usize = 0,

pub fn read(allocator: std.mem.Allocator, frame_type: Markers, quantization_tables: *[4]?QuantizationTable, dc_huffman_tables: *[4]?HuffmanTable, ac_huffman_tables: *[4]?HuffmanTable, reader: *std.Io.Reader) Image.ReadError!Frame {
    const frame_header = try FrameHeader.read(allocator, reader);

    const horizontal_sampling_factor_max = frame_header.getMaxHorizontalSamplingFactor();
    const vertical_sampling_factor_max = frame_header.getMaxVerticalSamplingFactor();

    const mcu_width = 8 * horizontal_sampling_factor_max;
    const mcu_height = 8 * vertical_sampling_factor_max;
    const width_actual = ((frame_header.width + mcu_width - 1) / mcu_width) * mcu_width;
    const height_actual = ((frame_header.height + mcu_height - 1) / mcu_height) * mcu_height;
    const block_storage = try allocator.alloc([MAX_COMPONENTS]Block, width_actual * height_actual / 64);
    // Unlike upstream zigimg we zero the grid: progressive scans only touch
    // the spectral bands they code, and uncoded coefficients must be 0.
    @memset(std.mem.sliceAsBytes(block_storage), 0);

    var self = Frame{
        .allocator = allocator,
        .frame_header = frame_header,
        .quantization_tables = quantization_tables,
        .dc_huffman_tables = dc_huffman_tables,
        .ac_huffman_tables = ac_huffman_tables,
        .frame_type = frame_type,
        .block_storage = block_storage,
        .block_height_actual = @intCast((height_actual + 7) / 8),
        .block_width_actual = @intCast((width_actual + 7) / 8),
        .block_height = (frame_header.height + 7) / 8,
        .block_width = (frame_header.width + 7) / 8,
        .horizontal_sampling_factor_max = horizontal_sampling_factor_max,
        .vertical_sampling_factor_max = vertical_sampling_factor_max,
    };
    errdefer self.deinit();

    return self;
}

pub fn deinit(self: *Frame) void {
    self.allocator.free(self.block_storage);
    for (self.dc_huffman_tables) |*maybe_huffman_table| {
        if (maybe_huffman_table.*) |*huffman_table| {
            huffman_table.deinit();
            maybe_huffman_table.* = null;
        }
    }

    for (self.ac_huffman_tables) |*maybe_huffman_table| {
        if (maybe_huffman_table.*) |*huffman_table| {
            huffman_table.deinit();
            maybe_huffman_table.* = null;
        }
    }

    self.frame_header.deinit();
}
