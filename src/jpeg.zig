//! JPEG coefficient reading — the Zig equivalent of jpeg2png's jpeg.c, which
//! used libjpeg's jpeg_read_coefficients. The entropy decoding lives in
//! src/jpeg/ (adapted from zigimg); this file converts the decoder's shared
//! MCU-padded block grid into jpeg2png's compact per-component layout and
//! ports decode_coefficients (dequantize + IDCT per block).
const std = @import("std");
const zigimg = @import("zigimg");

const Decoder = @import("jpeg/decoder.zig").Decoder;
const dct = @import("dct.zig");

/// Per-component DCT data, mirroring jpeg2png's `struct coef`.
/// `data` holds the raw quantized coefficients, block-major: consecutive
/// [64]i16 blocks in natural (de-zigzagged) order, blocks ordered row-major
/// over the component's libjpeg-style grid (width_in_blocks x
/// height_in_blocks, MCU padding cropped). `quant_table` is in natural order.
pub const Coef = struct {
    /// block-padded image height for this component (height_in_blocks * 8)
    h: u32,
    /// block-padded image width for this component (width_in_blocks * 8)
    w: u32,
    /// vertical subsampling factor (max_v_samp / comp_v_samp)
    h_samp: u32,
    /// horizontal subsampling factor (max_h_samp / comp_h_samp)
    w_samp: u32,
    /// quantized DCT coefficients
    data: []i16,
    /// image data (filled by decodeCoefficients / compute)
    fdata: []f32 = &.{},
    /// quantization table
    quant_table: [64]u16,
};

pub const Jpeg = struct {
    h: u32,
    w: u32,
    /// number of components (1-4: gray, rgb/yuv, cmyk); only coefs[0..ncomp]
    /// are valid
    ncomp: u32,
    /// true for progressive (SOF2) frames, false for baseline (SOF0)
    progressive: bool = false,
    coefs: [4]Coef,

    pub fn deinit(self: *Jpeg, gpa: std.mem.Allocator) void {
        for (self.coefs[0..self.ncomp]) |*coef| {
            gpa.free(coef.data);
            if (coef.fdata.len != 0) gpa.free(coef.fdata);
            coef.* = undefined;
        }
    }
};

pub const ReadCoefficientsError = zigimg.Image.ReadError || error{
    /// C fork: "only jpegs with 1 to 4 components are supported"
    InvalidComponentCount,
    /// C: "invalid quantization table" / "weird jpeg: no quant table pointer"
    InvalidQuantTable,
    /// C: "jpeg invalid coef h size" / "jpeg invalid coef w size"
    InvalidCoefSize,
    /// C: "jpeg is too big to fit in memory"
    ImageTooBig,
};

inline fn updiv(x: u32, y: u32) u32 {
    return (x + (y - 1)) / y;
}

/// Read the raw quantized DCT coefficients and quantization tables of a JPEG.
/// Equivalent of jpeg2png's read_jpeg().
pub fn readCoefficients(gpa: std.mem.Allocator, jpeg_bytes: []const u8) ReadCoefficientsError!Jpeg {
    var stream = zigimg.io.ReadStream.initMemory(jpeg_bytes);
    var decoder = Decoder.init(gpa);
    defer decoder.deinit();
    const frame = try decoder.read(&stream);

    const header = frame.frame_header;
    if (header.components.len < 1 or header.components.len > 4) return error.InvalidComponentCount;

    const jpeg_w: u32 = header.width;
    const jpeg_h: u32 = header.height;
    const h_max: u32 = @intCast(header.getMaxHorizontalSamplingFactor());
    const v_max: u32 = @intCast(header.getMaxVerticalSamplingFactor());

    var jpeg = Jpeg{
        .w = jpeg_w,
        .h = jpeg_h,
        .ncomp = @intCast(header.components.len),
        .progressive = frame.frame_type == .sof2,
        .coefs = undefined,
    };
    var initialized: usize = 0;
    errdefer for (jpeg.coefs[0..initialized]) |*coef| gpa.free(coef.data);

    for (header.components, 0..) |component, c| {
        // quantization table, widened to u16 (already in natural order)
        var quant: [64]u16 = undefined;
        switch (decoder.quantization_tables[component.quantization_table_id] orelse return error.InvalidQuantTable) {
            .q8 => |t| for (t, &quant) |v, *q| {
                q.* = v;
            },
            .q16 => |t| quant = t,
        }
        for (quant) |v| {
            if (v == 0) return error.InvalidQuantTable;
        }

        const h_c: u32 = component.horizontal_sampling_factor;
        const v_c: u32 = component.vertical_sampling_factor;
        // libjpeg's width_in_blocks / height_in_blocks
        const blocks_w = (jpeg_w * h_c + h_max * 8 - 1) / (h_max * 8);
        const blocks_h = (jpeg_h * v_c + v_max * 8 - 1) / (v_max * 8);
        const w = blocks_w * 8;
        const h = blocks_h * 8;
        const w_samp = h_max / h_c;
        const h_samp = v_max / v_c;

        // sanity checks ported from jpeg2png's read_jpeg, with the fork's
        // rounding fix (ceil of ceil instead of ceil of floor)
        if (h / 8 != updiv(updiv(jpeg_h, h_samp), 8)) return error.InvalidCoefSize;
        if (w / 8 != updiv(updiv(jpeg_w, w_samp), 8)) return error.InvalidCoefSize;
        const size_max: usize = std.math.maxInt(usize);
        if (size_max / h / w / h_samp / w_samp < 6) return error.ImageTooBig;

        const data = try gpa.alloc(i16, @as(usize, w) * h);
        errdefer gpa.free(data);

        // Gather this component's blocks from the shared MCU-padded grid,
        // dropping MCU-padding blocks (same crop libjpeg hands jpeg2png).
        var out: usize = 0;
        for (0..blocks_h) |by| {
            const grid_row = (by / v_c) * v_max + (by % v_c);
            for (0..blocks_w) |bx| {
                const grid_col = (bx / h_c) * h_max + (bx % h_c);
                const block = &frame.block_storage[grid_row * frame.block_width_actual + grid_col][c];
                for (block) |v| {
                    data[out] = std.math.cast(i16, v) orelse return error.InvalidData;
                    out += 1;
                }
            }
        }

        jpeg.coefs[c] = .{
            .h = h,
            .w = w,
            .h_samp = h_samp,
            .w_samp = w_samp,
            .data = data,
            .quant_table = quant,
        };
        initialized += 1;
    }

    return jpeg;
}

/// Decode DCT coefficients into image data (block layout). Literal port of
/// jpeg2png's decode_coefficients().
pub fn decodeCoefficients(gpa: std.mem.Allocator, coef: *Coef) error{OutOfMemory}!void {
    coef.fdata = try gpa.alloc(f32, @as(usize, coef.w) * coef.h);
    const blocks = @as(usize, coef.h / 8) * (coef.w / 8);
    for (0..blocks) |i| {
        for (0..64) |j| {
            // int multiply, then a single int->float rounding (as in C)
            coef.fdata[i * 64 + j] = @floatFromInt(@as(i32, coef.data[i * 64 + j]) * coef.quant_table[j]);
        }
        dct.idct8x8s(coef.fdata[i * 64 ..][0..64]);
    }
}
