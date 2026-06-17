//! zjpeg2png — Zig port of jpeg2png (https://github.com/victorvde/jpeg2png).
//! Decodes a JPEG into the smoothest picture that encodes to the same file,
//! by convex optimization (TV + TGV2 + DCT-coefficient deviation) instead of
//! filling missing information with decoding artifacts.
const std = @import("std");

pub const jpeg = @import("jpeg.zig");
pub const dct = @import("dct.zig");
pub const boxing = @import("box.zig");
pub const compute = @import("compute.zig");
pub const png = @import("png.zig");
pub const pipeline = @import("pipeline.zig");
pub const logger = @import("logger.zig");
pub const progressbar = @import("progressbar.zig");

test {
    std.testing.refAllDecls(@This());
    _ = @import("jpeg/utils.zig");
}
