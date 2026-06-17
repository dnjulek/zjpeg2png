//! Root `zjpeg2png` command: smooth-decode one or more JPEG files to PNG.
//! Built on zli; the decode pipeline itself lives in the zjpeg2png module and is
//! shared with the bit-exactness tests (which bypass the CLI entirely).
const std = @import("std");
const zli = @import("zli");
const zjpeg2png = @import("zjpeg2png");
const zon = @import("zon");

const options = @import("../options.zig");
const progress_mod = @import("progress.zig");
const version_cmd = @import("version.zig");
const info_cmd = @import("info.zig");

// zon.version is the build.zig.zon version, baked in at compile time.
pub const version_string = std.fmt.comptimePrint(
    "zjpeg2png {f} (Zig port of jpeg2png 1.01) — GPL-3.0-or-later",
    .{zon.version},
);

var active_progress: ?*progress_mod.DecodeProgress = null;

/// abort with a message on stderr, clearing the progress line first
/// (the same contract as jpeg2png's die())
pub fn die(comptime fmt: []const u8, args: anytype) noreturn {
    if (active_progress) |bar| {
        bar.clear();
        active_progress = null;
    }
    std.debug.print("zjpeg2png: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

const help_text =
    "Decodes JPEGs into the smoothest possible picture that encodes to the\n" ++
    "same file, by convex optimization (TV + TGV2) instead of filling the\n" ++
    "missing information with decoding artifacts.\n" ++
    "\n" ++
    "Examples:\n" ++
    "  zjpeg2png picture.jpg                 decode to picture.png\n" ++
    "  zjpeg2png picture.jpg -o out.png      explicit output name (overwrites)\n" ++
    "  zjpeg2png a.jpg b.jpg -O smooth/ -f   batch into a directory\n" ++
    "  zjpeg2png picture.jpg -i 500 -q       more iterations, no progress bar\n" ++
    "  zjpeg2png info picture.jpg            inspect JPEG metadata\n";

pub fn build(init_options: zli.InitOptions) !*zli.Command {
    const root = try zli.Command.init(init_options, .{
        .name = "zjpeg2png",
        .description = "Silky smooth JPEG decoding — no more artifacts",
        .help = help_text,
        .usage = "zjpeg2png <files...> [flags] | zjpeg2png <command>",
    }, run);

    try root.addFlags(&.{
        .{
            .name = "output",
            .shortcut = "o",
            .description = "Output file name (single input only; always overwrites)",
            .type = .String,
            .default_value = .{ .String = "" },
        },
        .{
            .name = "output-dir",
            .shortcut = "O",
            .description = "Write outputs into this directory (created if missing)",
            .type = .String,
            .default_value = .{ .String = "" },
        },
        .{
            .name = "force",
            .shortcut = "f",
            .description = "Overwrite existing output files",
            .type = .Bool,
            .default_value = .{ .Bool = false },
        },
        .{
            .name = "second-order-weight",
            .shortcut = "w",
            .description = "TGV weight; higher = smoother gradients, 0 = plain TV. Per-channel: w,wcb,wcr (needs -s)",
            .type = .String,
            .default_value = .{ .String = "0.3" },
        },
        .{
            .name = "probability-weight",
            .shortcut = "p",
            .description = "DCT distance weight; higher = closer to normal decoding, 0 = off. Per-channel: p,pcb,pcr",
            .type = .String,
            .default_value = .{ .String = "0.001" },
        },
        .{
            .name = "iterations",
            .shortcut = "i",
            .description = "Optimization steps; more = smoother but slower. Per-channel: n,ncb,ncr (needs -s)",
            .type = .String,
            .default_value = .{ .String = "50" },
        },
        .{
            .name = "quiet",
            .shortcut = "q",
            .description = "Don't show the progress bar",
            .type = .Bool,
            .default_value = .{ .Bool = false },
        },
        .{
            .name = "separate-components",
            .shortcut = "s",
            .description = "Optimize Y/Cb/Cr independently (faster, parallel; component edges may disagree)",
            .type = .Bool,
            .default_value = .{ .Bool = false },
        },
        .{
            .name = "threads",
            .shortcut = "t",
            .description = "Maximum threads; 0 = number of CPUs",
            .type = .Int,
            .default_value = .{ .Int = 0 },
        },
        .{
            .name = "16-bits-png",
            .shortcut = "1",
            .description = "Output 16-bit PNG (use many iterations with this)",
            .type = .Bool,
            .default_value = .{ .Bool = false },
        },
        .{
            .name = "csv-log",
            .shortcut = "c",
            .description = "Write per-iteration objective values to this CSV file",
            .type = .String,
            .default_value = .{ .String = "" },
        },
        .{
            .name = "version",
            .shortcut = "V",
            .description = "Show version information",
            .type = .Bool,
            .default_value = .{ .Bool = false },
        },
    });

    try root.addPositionalArg(.{
        .name = "files",
        .description = "JPEG files to decode",
        .required = false,
        .variadic = true,
    });

    try root.addCommands(&.{
        try version_cmd.register(init_options),
        try info_cmd.register(init_options),
    });

    return root;
}

fn stripJpegExt(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ".jpeg")) return name[0 .. name.len - 5];
    if (std.mem.endsWith(u8, name, ".jpg")) return name[0 .. name.len - 4];
    return name;
}

/// pre-flight checks for derived output names, ported from jpeg2png's main():
/// input must be openable; refuse to overwrite without --force; probe output
/// writability by creating and removing the file
fn checkAndProbeOutput(io: std.Io, infile: []const u8, outfile: []const u8, force: bool) void {
    const cwd = std.Io.Dir.cwd();
    {
        const in = cwd.openFile(io, infile, .{}) catch die("could not open input file `{s}`", .{infile});
        in.close(io);
    }
    if (!force) {
        if (cwd.openFile(io, outfile, .{})) |out| {
            out.close(io);
            die("not overwriting output file `{s}` (use --force)", .{outfile});
        } else |_| {}
    }
    const out = cwd.createFile(io, outfile, .{}) catch die("could not open output file `{s}`", .{outfile});
    out.close(io);
    cwd.deleteFile(io, outfile) catch {};
}

const FileJobs = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opts: zjpeg2png.pipeline.Options,
    inputs: []const []const u8,
    outputs: []const []const u8,
    progress: ?zjpeg2png.progressbar.Progress,
    bar: ?*progress_mod.DecodeProgress,
    base_log: zjpeg2png.logger.Logger,
    next: std.atomic.Value(usize) = .init(0),
};

fn fileWorker(jobs: *FileJobs) void {
    while (true) {
        const idx = jobs.next.fetchAdd(1, .monotonic);
        if (idx >= jobs.inputs.len) return;
        const infile = jobs.inputs[idx];
        const outfile = jobs.outputs[idx];
        var log = jobs.base_log;
        log.filename = infile;
        zjpeg2png.pipeline.decodeFile(jobs.gpa, jobs.io, infile, outfile, jobs.opts, jobs.progress, &log) catch |err| switch (err) {
            error.CouldNotOpenInput => die("could not open input file `{s}`", .{infile}),
            error.CouldNotWriteOutput => die("could not open output file `{s}`", .{outfile}),
            error.OutOfMemory => die("could not allocate memory", .{}),
            error.InvalidComponentCount => die("only jpegs with 1 to 4 components are supported (gray, rgb/yuv or cmyk)", .{}),
            error.InvalidQuantTable => die("invalid quantization table", .{}),
            error.InvalidCoefSize => die("jpeg invalid coef size", .{}),
            error.ImageTooBig => die("jpeg is too big to fit in memory", .{}),
            else => die("could not decode jpeg `{s}`: {t}", .{ infile, err }),
        };
        if (jobs.bar) |bar| {
            bar.fileDone(infile, outfile);
        }
    }
}

fn run(ctx: zli.CommandContext) !void {
    const gpa = ctx.allocator;
    const io = ctx.io;
    const stdout = ctx.writer;

    if (ctx.flag("version", bool)) {
        try stdout.print("{s}\n", .{version_string});
        try stdout.flush();
        return;
    }

    const files = ctx.positional_args;
    if (files.len == 0) {
        try ctx.command.printHelp();
        try stdout.flush();
        std.process.exit(1);
    }

    const separate = ctx.flag("separate-components", bool);
    const values = options.resolve(
        ctx.flag("second-order-weight", []const u8),
        ctx.flag("probability-weight", []const u8),
        ctx.flag("iterations", []const u8),
        separate,
    ) catch |err| switch (err) {
        error.InvalidWeight => die("invalid weight", .{}),
        error.WeightsRequireSeparate => die("different weights are only possible when using separated components (-s)", .{}),
        error.InvalidProbabilityWeight => die("invalid probability weight", .{}),
        error.InvalidIterations => die("invalid number of iterations", .{}),
        error.IterationsRequireSeparate => die("different iteration counts are only possible when using separated components (-s)", .{}),
    };

    const force = ctx.flag("force", bool);
    const quiet = ctx.flag("quiet", bool);
    const sixteen_bit = ctx.flag("16-bits-png", bool);
    const out_flag = ctx.flag("output", []const u8);
    const outdir_flag = ctx.flag("output-dir", []const u8);
    const csv_path = ctx.flag("csv-log", []const u8);

    const threads_flag = ctx.flag("threads", i32);
    if (threads_flag < 0) die("invalid number of threads", .{});
    const threads: u32 = if (threads_flag == 0)
        @intCast(@max(1, std.Thread.getCpuCount() catch 1))
    else
        @intCast(threads_flag);
    // the optimizer is memory-bandwidth bound: beyond ~8 threads a single
    // image gets slower, so the auto default caps compute threads while an
    // explicit -t is honored as given
    const compute_threads: u32 = if (threads_flag == 0) @min(threads, 8) else threads;

    const nin = files.len;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // resolve output file names
    if (out_flag.len != 0 and outdir_flag.len != 0) {
        die("--output and --output-dir are mutually exclusive", .{});
    }
    if (out_flag.len != 0 and nin != 1) {
        die("--output works with exactly one input file; use --output-dir for batches", .{});
    }

    const cwd = std.Io.Dir.cwd();
    const outfiles = try arena.alloc([]const u8, nin);
    if (out_flag.len != 0) {
        outfiles[0] = out_flag;
    } else if (outdir_flag.len != 0) {
        cwd.createDirPath(io, outdir_flag) catch die("could not create output directory `{s}`", .{outdir_flag});
        for (files, 0..) |infile, idx| {
            const stem = stripJpegExt(std.fs.path.basename(infile));
            const name = try std.fmt.allocPrint(arena, "{s}.png", .{stem});
            outfiles[idx] = try std.fs.path.join(arena, &.{ outdir_flag, name });
            checkAndProbeOutput(io, infile, outfiles[idx], force);
        }
    } else {
        for (files, 0..) |infile, idx| {
            outfiles[idx] = try std.fmt.allocPrint(arena, "{s}.png", .{stripJpegExt(infile)});
            checkAndProbeOutput(io, infile, outfiles[idx], force);
        }
    }

    // csv logger
    var base_log = zjpeg2png.logger.Logger{};
    var csv_buffer: [4096]u8 = undefined;
    var csv_writer: std.Io.File.Writer = undefined;
    var csv_sink: zjpeg2png.logger.CsvSink = undefined;
    var csv_file: ?std.Io.File = null;
    if (csv_path.len != 0) {
        const file = cwd.createFile(io, csv_path, .{}) catch die("could not open csv log `{s}`", .{csv_path});
        csv_file = file;
        csv_writer = file.writer(io, &csv_buffer);
        csv_sink = .{ .writer = &csv_writer.interface, .io = io };
        csv_sink.writeHeader() catch die("could not write to csv log", .{});
        base_log.sink = &csv_sink;
    }

    // progress bar
    var progress_storage: progress_mod.DecodeProgress = undefined;
    var bar: ?*progress_mod.DecodeProgress = null;
    if (!quiet) {
        const total: u64 = if (!separate)
            @as(u64, nin) * values.iterations[0]
        else
            @as(u64, nin) * (@as(u64, values.iterations[0]) + values.iterations[1] + values.iterations[2]);
        progress_storage = progress_mod.DecodeProgress.start(stdout, io, total, @intCast(nin));
        bar = &progress_storage;
        active_progress = bar;
    }

    var jobs = FileJobs{
        .gpa = gpa,
        .io = io,
        .opts = .{
            .iterations = values.iterations,
            .weights = values.weights,
            .pweights = values.pweights,
            .png_bits = if (sixteen_bit) 16 else 8,
            .all_together = !separate,
            // channel parallelism only when not already parallel over files;
            // non-3-component files always use the per-channel path, so give
            // them threads too
            .channel_threads = if (nin == 1) @min(4, threads) else 1,
            // threads inside one optimization (joint mode / single channel)
            .compute_threads = if (nin == 1) compute_threads else 1,
        },
        .inputs = files,
        .outputs = outfiles,
        .progress = if (bar) |b| b.progress() else null,
        .bar = bar,
        .base_log = base_log,
    };

    // decode each file smoothly, in parallel when multiple files are given;
    // the main thread acts as one of the workers
    const nthreads: usize = @min(threads, nin);
    if (nthreads <= 1) {
        fileWorker(&jobs);
    } else {
        const workers = try arena.alloc(?std.Thread, nthreads - 1);
        for (workers) |*worker| {
            worker.* = std.Thread.spawn(.{}, fileWorker, .{&jobs}) catch null;
        }
        fileWorker(&jobs);
        for (workers) |worker| {
            if (worker) |thread| thread.join();
        }
    }

    if (bar) |b| {
        b.finish(outfiles);
        active_progress = null;
    }
    if (csv_file) |file| {
        csv_writer.interface.flush() catch die("could not write to csv log", .{});
        file.close(io);
    }
    try stdout.flush();
}
