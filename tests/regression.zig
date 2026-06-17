//! Regression / determinism guard for the decode pipeline.
//!
//! Decoding a JPEG must always produce the exact same pixels. This test embeds
//! `tests/image.jpg` (512x512, 3-component YCbCr — the joint-optimization path)
//! so there is no cwd or filesystem dependency, runs the full pipeline on the
//! bit-exact serial path, and pins a SHA-256 of the rendered RGB to a golden
//! digest. Any change in the optimizer's output — intended or not — flips the
//! hash and fails here.
//!
//! If you change the output *on purpose*, run the test once, copy the printed
//! `actual` digest into `golden_sha256`, and commit it.
const std = @import("std");
const zjpeg2png = @import("zjpeg2png");

const Sha256 = std.crypto.hash.sha2.Sha256;

const image_jpg = @embedFile("image.jpg");

// Fixed config. `compute_threads = 1` / `channel_threads = 1` select the
// bit-exact serial path, which is fully deterministic and identical in Debug
// and ReleaseFast (the project never enables float contraction). Weights and
// the probability term stay at their defaults so TV + TGV2 + prob all run.
const opts = zjpeg2png.pipeline.Options{
    .iterations = .{ 16, 16, 16, 16 },
    .compute_threads = 1,
    .channel_threads = 1,
};

// SHA-256 (lowercase hex) of the 512*512*3 rendered RGB bytes for `opts`.
const golden_sha256 = "d33261816346469a29c3e7b6267a19e7752732ab43ba3a5f941eb58624641bfd";

fn renderDigest(gpa: std.mem.Allocator) ![Sha256.digest_length]u8 {
    const log = zjpeg2png.logger.Logger{};
    var rendered = try zjpeg2png.pipeline.decodeToPixels(gpa, image_jpg, opts, null, &log);
    defer rendered.deinit(gpa);
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(rendered.rgb8, &digest, .{});
    return digest;
}

test "image.jpg decodes to the pinned golden pixels" {
    const digest = try renderDigest(std.testing.allocator);
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, golden_sha256)) {
        std.debug.print(
            \\
            \\regression: tests/image.jpg rendered to unexpected pixels
            \\  actual   = {s}
            \\  expected = {s}
            \\
        , .{ hex, golden_sha256 });
        return error.PixelRegression;
    }
}

test "decoding image.jpg is deterministic across runs" {
    const a = try renderDigest(std.testing.allocator);
    const b = try renderDigest(std.testing.allocator);
    try std.testing.expectEqualSlices(u8, &a, &b);
}
