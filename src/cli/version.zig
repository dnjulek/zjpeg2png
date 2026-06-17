//! `zjpeg2png version` subcommand.
const zli = @import("zli");

const cli_root = @import("root.zig");

pub fn register(init_options: zli.InitOptions) !*zli.Command {
    return zli.Command.init(init_options, .{
        .name = "version",
        .description = "Show version information",
    }, run);
}

fn run(ctx: zli.CommandContext) !void {
    try ctx.writer.print("{s}\n", .{cli_root.version_string});
}
