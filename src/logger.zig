//! CSV optimization logger, port of jpeg2png's logger.c. The sink is shared
//! between threads (one Logger value per compute call, copied like the C
//! code's firstprivate(log)).
const std = @import("std");

pub const CsvSink = struct {
    writer: *std.Io.Writer,
    io: std.Io,
    mutex: std.Io.Mutex = .init,

    pub fn writeHeader(self: *CsvSink) !void {
        try self.writer.writeAll("filename,channel,iteration,objective,prob_dist,tv,tv2\n");
    }
};

pub const Logger = struct {
    sink: ?*CsvSink = null,
    filename: []const u8 = "",
    channel: u32 = 0,
    iteration: u32 = 0,

    pub fn log(self: *Logger, objective: f64, prob_dist: f64, tv: f64, tv2: f64) void {
        const sink = self.sink orelse return;
        sink.mutex.lockUncancelable(sink.io);
        defer sink.mutex.unlock(sink.io);
        sink.writer.print("{s},{d},{d},{d:.6},{d:.6},{d:.6},{d:.6}\n", .{
            self.filename, self.channel, self.iteration, objective, prob_dist, tv, tv2,
        }) catch {
            std.debug.print("jpeg2png: could not write to csv log\n", .{});
            std.process.exit(1);
        };
    }
};
