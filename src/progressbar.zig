//! Progress reporting. `Progress` is the type-erased sink the optimizer
//! ticks once per iteration; `ProgressBar` is the classic renderer ported
//! from jpeg2png's progressbar.c (70-char bar + percent, redrawn only when
//! the displayed value changes). The CLI plugs in its own fancier renderer.
const std = @import("std");

/// Type-erased progress sink; compute() calls inc() once per iteration.
pub const Progress = struct {
    context: *anyopaque,
    incFn: *const fn (context: *anyopaque) void,

    pub fn inc(self: Progress) void {
        self.incFn(self.context);
    }
};

const width = 70;

pub const ProgressBar = struct {
    current: u32 = 0,
    max: u32,
    out: *std.Io.Writer,
    io: std.Io,
    mutex: std.Io.Mutex = .init,

    pub fn start(out: *std.Io.Writer, io: std.Io, max: u32) ProgressBar {
        var pb = ProgressBar{ .max = @max(max, 1), .out = out, .io = io };
        pb.show();
        return pb;
    }

    fn toPrint(self: *const ProgressBar, current: u32) u32 {
        return width * current / self.max;
    }

    fn percentage(self: *const ProgressBar, current: u32) u32 {
        return 100 * current / self.max;
    }

    fn show(self: *ProgressBar) void {
        const to_print = self.toPrint(self.current);
        self.out.writeAll("\r[") catch return;
        self.out.splatByteAll('#', to_print) catch return;
        self.out.splatByteAll(' ', width - to_print) catch return;
        self.out.print("] {d: >3}%", .{self.percentage(self.current)}) catch return;
        self.out.flush() catch return;
    }

    fn setLocked(self: *ProgressBar, current: u32) void {
        const old_to_print = self.toPrint(self.current);
        const old_percentage = self.percentage(self.current);
        const to_print = self.toPrint(current);
        const percent = self.percentage(current);
        self.current = current;
        if (old_to_print == to_print and old_percentage == percent) {
            return;
        }
        self.show();
    }

    pub fn set(self: *ProgressBar, current: u32) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.setLocked(current);
    }

    pub fn inc(self: *ProgressBar) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.setLocked(self.current + 1);
    }

    /// Adapt to the type-erased Progress interface.
    pub fn progress(self: *ProgressBar) Progress {
        return .{ .context = self, .incFn = &incOpaque };
    }

    fn incOpaque(context: *anyopaque) void {
        const self: *ProgressBar = @ptrCast(@alignCast(context));
        self.inc();
    }

    pub fn clear(self: *ProgressBar) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.out.writeAll("\r") catch return;
        self.out.splatByteAll(' ', width + 7) catch return;
        self.out.flush() catch return;
        self.out.writeAll("\r") catch return;
        self.out.flush() catch return;
    }
};
