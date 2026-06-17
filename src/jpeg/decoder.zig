//! Adapted from zigimg (https://github.com/zigimg/zigimg), MIT license.
//! Source: the JPEG marker loop of src/formats/jpeg.zig, modified to STOP
//! after entropy decoding: no dequantization, no IDCT, no pixel rendering.
//! After read() returns, frame.block_storage holds the raw quantized DCT
//! coefficients exactly as stored in the file (de-zigzagged, DC prediction
//! applied), which is what jpeg2png needs.
const std = @import("std");

const zigimg = @import("zigimg");
const Image = zigimg.Image;
const io = zigimg.io;

const Markers = @import("utils.zig").Markers;
const QuantizationTable = @import("quantization.zig").Table;
const HuffmanTable = @import("huffman.zig").Table;
const Frame = @import("Frame.zig");
const Scan = @import("Scan.zig");

pub const Decoder = struct {
    frame: ?Frame = null,
    allocator: std.mem.Allocator,
    quantization_tables: [4]?QuantizationTable = @splat(null),
    dc_huffman_tables: [4]?HuffmanTable = @splat(null),
    ac_huffman_tables: [4]?HuffmanTable = @splat(null),
    restart_interval: u16 = 0,

    pub fn init(allocator: std.mem.Allocator) Decoder {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Decoder) void {
        if (self.frame) |*frame| {
            // Frame.deinit also releases the huffman tables (it points at ours).
            frame.deinit();
        } else {
            for (&self.dc_huffman_tables) |*maybe_table| {
                if (maybe_table.*) |*table| table.deinit();
            }
            for (&self.ac_huffman_tables) |*maybe_table| {
                if (maybe_table.*) |*table| table.deinit();
            }
        }
    }

    fn parseDefineQuantizationTables(self: *Decoder, reader: *std.Io.Reader) Image.ReadError!void {
        var segment_size = try reader.takeInt(u16, .big);
        segment_size -= 2;

        while (segment_size > 0) {
            const precision_and_destination = try reader.takeByte();
            const table_precision = precision_and_destination >> 4;
            const table_destination = precision_and_destination & 0b11;

            const quantization_table = try QuantizationTable.read(table_precision, reader);
            switch (quantization_table) {
                .q8 => segment_size -= 64 + 1,
                .q16 => segment_size -= 128 + 1,
            }

            self.quantization_tables[table_destination] = quantization_table;
        }
    }

    fn parseDefineHuffmanTables(self: *Decoder, reader: *std.Io.Reader) Image.ReadError!void {
        var segment_size = try reader.takeInt(u16, .big);
        segment_size -= 2;

        while (segment_size > 0) {
            const class_and_destination = try reader.takeByte();
            const table_class = class_and_destination >> 4;
            const table_destination = class_and_destination & 0x0F;

            const huffman_table = try HuffmanTable.read(self.allocator, table_class, reader);

            if (table_class == 0) {
                if (self.dc_huffman_tables[table_destination]) |*old_huffman_table| {
                    old_huffman_table.deinit();
                }
                self.dc_huffman_tables[table_destination] = huffman_table;
            } else {
                if (self.ac_huffman_tables[table_destination]) |*old_huffman_table| {
                    old_huffman_table.deinit();
                }
                self.ac_huffman_tables[table_destination] = huffman_table;
            }

            // Class+Destination + code counts + code table
            segment_size -= 1 + 16 + @as(u16, @intCast(huffman_table.code_map.count()));
        }
    }

    fn parseDefineRestartInterval(self: *Decoder, reader: *std.Io.Reader) Image.ReadError!void {
        const segment_length = try reader.takeInt(u16, .big);
        if (segment_length != 4) return Image.ReadError.InvalidData;

        self.restart_interval = try reader.takeInt(u16, .big);
    }

    fn parseScan(self: *Decoder, read_stream: *io.ReadStream) Image.ReadError!void {
        if (self.frame) |frame| {
            try Scan.performScan(&frame, self.restart_interval, read_stream);
        } else {
            return Image.ReadError.InvalidData;
        }
    }

    /// Parse the whole stream up to (and including) EOI. The returned frame
    /// (owned by the decoder) holds the raw quantized coefficients.
    pub fn read(self: *Decoder, read_stream: *io.ReadStream) Image.ReadError!*Frame {
        const reader = read_stream.reader();
        var marker = try reader.takeInt(u16, .big);

        if (marker != @intFromEnum(Markers.start_of_image)) {
            return Image.ReadError.InvalidData;
        }

        while (marker != @intFromEnum(Markers.end_of_image)) {
            marker = try reader.takeInt(u16, .big);

            switch (std.enums.fromInt(Markers, marker) orelse return Image.ReadError.InvalidData) {
                .sof0, .sof2 => |sof| { // Baseline DCT, progressive DCT Huffman coding
                    if (self.frame != null) {
                        return Image.ReadError.Unsupported;
                    }

                    self.frame = try Frame.read(self.allocator, sof, &self.quantization_tables, &self.dc_huffman_tables, &self.ac_huffman_tables, reader);
                },

                // Unsupported SOF types (extended sequential, lossless,
                // differential, arithmetic coding) — same set as zigimg.
                .sof1, .sof3, .sof5, .sof6, .sof7, .sof9, .sof10, .sof11, .sof13, .sof14, .sof15 => return Image.ReadError.Unsupported,

                .define_huffman_tables => {
                    try self.parseDefineHuffmanTables(reader);
                },

                .start_of_scan => {
                    try self.parseScan(read_stream);
                },

                .define_quantization_tables => {
                    try self.parseDefineQuantizationTables(reader);
                },

                .comment => {
                    const comment_length = try reader.takeInt(u16, .big);
                    if (comment_length < 2) return Image.ReadError.InvalidData;
                    try read_stream.seekBy(comment_length - 2);
                },

                .app0, .app1, .app2, .app3, .app4, .app5, .app6, .app7, .app8, .app9, .app10, .app11, .app12, .app13, .app14, .app15 => {
                    const application_data_length = try reader.takeInt(u16, .big);
                    if (application_data_length < 2) return Image.ReadError.InvalidData;
                    try read_stream.seekBy(application_data_length - 2);
                },

                .define_restart_interval => {
                    try self.parseDefineRestartInterval(reader);
                },

                .restart0, .restart1, .restart2, .restart3, .restart4, .restart5, .restart6, .restart7 => {
                    continue;
                },

                .end_of_image => {
                    continue;
                },

                else => {
                    return Image.ReadError.InvalidData;
                },
            }
        }

        if (self.frame) |*frame| {
            return frame;
        }

        return Image.ReadError.InvalidData;
    }
};
