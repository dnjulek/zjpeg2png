//! Option *value* semantics ported from jpeg2png.c's main(): -w/-p/-i accept
//! a single value or a comma-separated per-channel triple (Y,Cb,Cr); triples
//! for -w/-i require separate-components mode. Tokenizing is zli's job now —
//! this module only interprets the value strings.
//!
//! Arrays have a 4th slot for the K channel of CMYK files. The let-def fork
//! reads it out of bounds (UB); we define it: -p/-i values for channel 3
//! mirror channel 0, the channel-3 weight is always 0 (chroma-like). The CLI
//! accepts at most 3 comma values, same as C.
const std = @import("std");

pub const Values = struct {
    weights: [4]f32 = .{ 0.3, 0, 0, 0 },
    pweights: [4]f32 = .{ 0.001, 0.001, 0.001, 0.001 },
    iterations: [4]u32 = .{ 50, 50, 50, 50 },
};

pub const ResolveError = error{
    InvalidWeight,
    WeightsRequireSeparate,
    InvalidProbabilityWeight,
    InvalidIterations,
    IterationsRequireSeparate,
};

/// Parse a comma-separated float list of at most 3 entries; null on any
/// invalid entry (stricter than C's sscanf, which ignored trailing garbage).
fn parseFloatList(s: []const u8, out: *[3]f32) ?usize {
    var it = std.mem.splitScalar(u8, s, ',');
    var n: usize = 0;
    while (it.next()) |part| {
        if (n == 3) return null;
        out[n] = std.fmt.parseFloat(f32, part) catch return null;
        n += 1;
    }
    return n;
}

fn parseIntList(s: []const u8, out: *[3]u32) ?usize {
    var it = std.mem.splitScalar(u8, s, ',');
    var n: usize = 0;
    while (it.next()) |part| {
        if (n == 3) return null;
        out[n] = std.fmt.parseInt(u32, part, 10) catch return null;
        n += 1;
    }
    return n;
}

/// Resolve the -w/-p/-i value strings (zli supplies the defaults "0.3",
/// "0.001", "50" when the flags are absent).
pub fn resolve(
    weight_s: []const u8,
    pweight_s: []const u8,
    iterations_s: []const u8,
    separate: bool,
) ResolveError!Values {
    var values = Values{};

    {
        var vals: [3]f32 = undefined;
        switch (parseFloatList(weight_s, &vals) orelse 0) {
            3 => {
                if (!separate) return error.WeightsRequireSeparate;
                values.weights = .{ vals[0], vals[1], vals[2], 0 }; // channel-3 weight stays 0
            },
            1 => values.weights[0] = vals[0], // chroma weights stay 0
            else => return error.InvalidWeight,
        }
    }
    {
        var vals: [3]f32 = undefined;
        switch (parseFloatList(pweight_s, &vals) orelse 0) {
            3 => values.pweights = .{ vals[0], vals[1], vals[2], vals[0] }, // channel 3 mirrors channel 0
            1 => values.pweights = @splat(vals[0]),
            else => return error.InvalidProbabilityWeight,
        }
    }
    {
        var vals: [3]u32 = undefined;
        switch (parseIntList(iterations_s, &vals) orelse 0) {
            3 => {
                if (!separate) return error.IterationsRequireSeparate;
                values.iterations = .{ vals[0], vals[1], vals[2], vals[0] }; // channel 3 mirrors channel 0
            },
            1 => values.iterations = @splat(vals[0]),
            else => return error.InvalidIterations,
        }
    }

    return values;
}

const testing = std.testing;

test "defaults resolve to jpeg2png defaults" {
    const v = try resolve("0.3", "0.001", "50", false);
    try testing.expectEqual(@as(f32, 0.3), v.weights[0]);
    try testing.expectEqual(@as(f32, 0), v.weights[1]);
    try testing.expectEqual(@as(f32, 0.001), v.pweights[2]);
    try testing.expectEqual(@as(u32, 50), v.iterations[1]);
}

test "single values broadcast (except chroma weights)" {
    const v = try resolve("0.5", "0.01", "25", false);
    try testing.expectEqual(@as(f32, 0.5), v.weights[0]);
    try testing.expectEqual(@as(f32, 0), v.weights[2]); // chroma weights stay 0
    try testing.expectEqual(@as(f32, 0), v.weights[3]);
    try testing.expectEqual(@as(f32, 0.01), v.pweights[1]);
    try testing.expectEqual(@as(f32, 0.01), v.pweights[3]);
    try testing.expectEqual(@as(u32, 25), v.iterations[2]);
    try testing.expectEqual(@as(u32, 25), v.iterations[3]);
}

test "triples require separate for weights and iterations but not pweights" {
    try testing.expectError(error.WeightsRequireSeparate, resolve("1,2,3", "0.001", "50", false));
    try testing.expectError(error.IterationsRequireSeparate, resolve("0.3", "0.001", "4,5,6", false));
    const a = try resolve("1,2,3", "0.1,0.2,0.3", "4,5,6", true);
    try testing.expectEqual(@as(f32, 2), a.weights[1]);
    try testing.expectEqual(@as(f32, 0.2), a.pweights[1]);
    try testing.expectEqual(@as(u32, 6), a.iterations[2]);
    // channel 3 (K of CMYK): weight stays 0, -p/-i mirror channel 0
    try testing.expectEqual(@as(f32, 0), a.weights[3]);
    try testing.expectEqual(@as(f32, 0.1), a.pweights[3]);
    try testing.expectEqual(@as(u32, 4), a.iterations[3]);
    const b = try resolve("0.3", "0.1,0.2,0.3", "50", false);
    try testing.expectEqual(@as(f32, 0.3), b.pweights[2]);
}

test "invalid values" {
    try testing.expectError(error.InvalidWeight, resolve("a", "0.001", "50", false));
    try testing.expectError(error.InvalidWeight, resolve("1,2", "0.001", "50", true));
    try testing.expectError(error.InvalidWeight, resolve("1,2,3,4", "0.001", "50", true));
    try testing.expectError(error.InvalidProbabilityWeight, resolve("0.3", "x", "50", false));
    try testing.expectError(error.InvalidIterations, resolve("0.3", "0.001", "1,2", true));
    try testing.expectError(error.InvalidIterations, resolve("0.3", "0.001", "-5", false));
}
