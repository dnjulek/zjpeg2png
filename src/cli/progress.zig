//! Rich progress rendering for the CLI, drawn with zli's styles: an animated
//! braille spinner, a smooth sub-character unicode bar, percentage, iteration
//! and file counters, and an ETA.
//!
//! Rendering is driven by inc() under our own mutex instead of zli's Spinner
//! background thread: the Spinner's updateMessage() frees the message string
//! while its render thread reads it unlocked, which is unsafe at the update
//! rates the optimizer produces. Same look, no race.
const std = @import("std");
const Io = std.Io;
const zli = @import("zli");
const zjpeg2png = @import("zjpeg2png");

const styles = zli.styles;
const spinner_frames: []const []const u8 = zli.SpinnerStyles.dots;

const bar_cells = 28;
// eighth-block partials for a smooth bar edge
const partials = [_][]const u8{ "", "▏", "▎", "▍", "▌", "▋", "▊", "▉" };
const clear_line = "\r\x1b[2K";

pub const DecodeProgress = struct {
    writer: *Io.Writer,
    io: Io,
    mutex: Io.Mutex = .init,
    iterations_done: u64 = 0,
    iterations_total: u64,
    files_done: u32 = 0,
    files_total: u32,
    start_time: Io.Timestamp,
    frame: usize = 0,
    last_render_ms: i64 = std.math.minInt(i64),

    pub fn start(writer: *Io.Writer, io: Io, iterations_total: u64, files_total: u32) DecodeProgress {
        var self = DecodeProgress{
            .writer = writer,
            .io = io,
            .iterations_total = @max(iterations_total, 1),
            .files_total = files_total,
            .start_time = Io.Timestamp.now(io, .awake),
        };
        self.render();
        return self;
    }

    /// Adapt to the library's type-erased Progress interface.
    pub fn progress(self: *DecodeProgress) zjpeg2png.progressbar.Progress {
        return .{ .context = self, .incFn = &incOpaque };
    }

    fn incOpaque(context: *anyopaque) void {
        const self: *DecodeProgress = @ptrCast(@alignCast(context));
        self.inc();
    }

    fn elapsedMs(self: *const DecodeProgress) i64 {
        const now = Io.Timestamp.now(self.io, .awake);
        return self.start_time.durationTo(now).toMilliseconds();
    }

    /// one optimizer iteration finished (called from worker threads)
    pub fn inc(self: *DecodeProgress) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.iterations_done += 1;
        // throttle redraws; always draw the final state
        const now_ms = self.elapsedMs();
        if (self.iterations_done < self.iterations_total and now_ms - self.last_render_ms < 80) {
            return;
        }
        self.render();
    }

    /// one input file fully decoded and written (batch mode prints a ✔ line)
    pub fn fileDone(self: *DecodeProgress, infile: []const u8, outfile: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.files_done += 1;
        if (self.files_total > 1) {
            self.writer.print(clear_line ++ "{s}✔{s} {s} {s}→{s} {s}\n", .{
                styles.GREEN, styles.RESET, infile,
                styles.DIM,   styles.RESET, outfile,
            }) catch return;
            self.render();
        }
    }

    /// erase the progress line (used before dying with an error)
    pub fn clear(self: *DecodeProgress) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.writer.writeAll(clear_line) catch return;
        self.writer.flush() catch return;
    }

    /// final ✔ summary line
    pub fn finish(self: *DecodeProgress, outfiles: []const []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const ms = self.elapsedMs();
        const secs: u64 = @intCast(@divTrunc(ms, 1000));
        const tenths: u64 = @intCast(@mod(@divTrunc(ms, 100), 10));
        self.writer.writeAll(clear_line) catch return;
        if (outfiles.len == 1) {
            self.writer.print("{s}✔{s} wrote {s}{s}{s} in {d}.{d}s\n", .{
                styles.GREEN, styles.RESET, styles.BOLD, outfiles[0], styles.RESET, secs, tenths,
            }) catch return;
        } else {
            self.writer.print("{s}✔{s} wrote {s}{d} files{s} in {d}.{d}s\n", .{
                styles.GREEN, styles.RESET, styles.BOLD, outfiles.len, styles.RESET, secs, tenths,
            }) catch return;
        }
        self.writer.flush() catch return;
    }

    fn render(self: *DecodeProgress) void {
        const done = @min(self.iterations_done, self.iterations_total);
        const total = self.iterations_total;
        const percent = 100 * done / total;

        // bar fill in eighths of a cell
        const eighths: usize = @intCast(done * bar_cells * 8 / total);
        const full = eighths / 8;
        const partial = eighths % 8;
        const rest = bar_cells - full - @intFromBool(partial != 0);

        const w = self.writer;
        w.writeAll(clear_line) catch return;
        w.print("{s}{s}{s}", .{ styles.CYAN, spinner_frames[self.frame], styles.RESET }) catch return;
        self.frame = (self.frame + 1) % spinner_frames.len;

        w.writeAll(styles.CYAN) catch return;
        w.splatBytesAll("█", full) catch return;
        w.writeAll(partials[partial]) catch return;
        w.writeAll(styles.RESET ++ styles.DIM) catch return;
        w.splatBytesAll("░", rest) catch return;
        w.writeAll(styles.RESET) catch return;

        w.print(" {s}{d: >3}%{s}", .{ styles.BOLD, percent, styles.RESET }) catch return;
        w.print("{s} · {d}/{d} iterations", .{ styles.DIM, done, total }) catch return;
        if (self.files_total > 1) {
            w.print(" · {d}/{d} files", .{ self.files_done, self.files_total }) catch return;
        }

        const ms = self.elapsedMs();
        if (done > 0 and done < total and ms > 500) {
            const eta_ms = @divTrunc(ms * @as(i64, @intCast(total - done)), @as(i64, @intCast(done)));
            const eta_s = @divTrunc(eta_ms, 1000);
            if (eta_s >= 60) {
                w.print(" · eta {d}m{d:0>2}s", .{ @divTrunc(eta_s, 60), @mod(eta_s, 60) }) catch return;
            } else {
                w.print(" · eta {d}s", .{eta_s}) catch return;
            }
        }
        w.writeAll(styles.RESET) catch return;
        w.flush() catch return;

        self.last_render_ms = ms;
    }
};
