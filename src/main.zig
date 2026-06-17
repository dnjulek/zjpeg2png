//! zjpeg2png CLI entry point — zli bootstrap; the commands live in src/cli/.
const std = @import("std");
const Io = std.Io;

const cli_root = @import("cli/root.zig");

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = Io.File.Writer.init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};

    var stdin_buffer: [256]u8 = undefined;
    var stdin_reader = Io.File.Reader.init(.stdin(), io, &stdin_buffer);
    const stdin = &stdin_reader.interface;

    const root = try cli_root.build(.{
        .allocator = init.gpa,
        .io = io,
        .writer = stdout,
        .reader = stdin,
    });
    defer root.deinit();

    var args_iter = try init.minimal.args.iterateAllocator(init.arena.allocator());
    defer args_iter.deinit();
    try root.execute(&args_iter, .{});
}

test {
    _ = @import("cli/root.zig");
    _ = @import("cli/version.zig");
    _ = @import("cli/info.zig");
    _ = @import("cli/progress.zig");
    _ = @import("options.zig");
}
