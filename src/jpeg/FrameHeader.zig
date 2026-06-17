//! Adapted from zigimg (https://github.com/zigimg/zigimg), MIT license.
//! Source: src/formats/jpeg/FrameHeader.zig — t-81 section B.2.2 frame header.
const std = @import("std");

const Image = @import("zigimg").Image;

pub const Component = struct {
    id: u8,
    horizontal_sampling_factor: u4,
    vertical_sampling_factor: u4,
    quantization_table_id: u8,

    pub fn read(reader: *std.Io.Reader) Image.ReadError!Component {
        const component_id = try reader.takeByte();
        const sampling_factors = try reader.takeByte();
        const quantization_table_id = try reader.takeByte();

        const horizontal_sampling_factor: u4 = @intCast(sampling_factors >> 4);
        const vertical_sampling_factor: u4 = @intCast(sampling_factors & 0xF);

        if (horizontal_sampling_factor < 1 or horizontal_sampling_factor > 4) {
            return Image.ReadError.InvalidData;
        }

        if (vertical_sampling_factor < 1 or vertical_sampling_factor > 4) {
            return Image.ReadError.InvalidData;
        }

        if (quantization_table_id > 3) {
            return Image.ReadError.InvalidData;
        }

        return Component{
            .id = component_id,
            .horizontal_sampling_factor = horizontal_sampling_factor,
            .vertical_sampling_factor = vertical_sampling_factor,
            .quantization_table_id = quantization_table_id,
        };
    }
};

const FrameHeader = @This();

allocator: std.mem.Allocator,
sample_precision: u8,
height: u16,
width: u16,
components: []Component,

pub fn read(allocator: std.mem.Allocator, reader: *std.Io.Reader) Image.ReadError!FrameHeader {
    const segment_size = try reader.takeInt(u16, .big);

    const sample_precision = try reader.takeByte();
    const height = try reader.takeInt(u16, .big);
    const width = try reader.takeInt(u16, .big);

    const component_count = try reader.takeByte();

    if (component_count < 1 or component_count > 4) {
        return Image.ReadError.InvalidData;
    }

    var components = try allocator.alloc(Component, component_count);
    errdefer allocator.free(components);

    var i: usize = 0;
    while (i < component_count) : (i += 1) {
        components[i] = try Component.read(reader);
    }

    // see B 8.2 table for the meaning of this check.
    std.debug.assert(segment_size == 8 + 3 * component_count);

    return FrameHeader{
        .allocator = allocator,
        .sample_precision = sample_precision,
        .height = height,
        .width = width,
        .components = components,
    };
}

pub fn deinit(self: *FrameHeader) void {
    self.allocator.free(self.components);
}

pub fn getMaxHorizontalSamplingFactor(self: FrameHeader) usize {
    var ret: u4 = 0;
    for (self.components) |component| {
        if (ret < component.horizontal_sampling_factor) {
            ret = component.horizontal_sampling_factor;
        }
    }

    return ret;
}

pub fn getMaxVerticalSamplingFactor(self: FrameHeader) usize {
    var ret: u4 = 0;
    for (self.components) |component| {
        if (ret < component.vertical_sampling_factor) {
            ret = component.vertical_sampling_factor;
        }
    }

    return ret;
}
