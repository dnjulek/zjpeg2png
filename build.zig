const std = @import("std");
const zon = @import("build.zig.zon");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const options = b.addOptions();
    options.addOption(std.SemanticVersion, "version", try std.SemanticVersion.parse(zon.version));

    const zigimg_dependency = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    const zigimg_module = zigimg_dependency.module("zigimg");

    const zli_dependency = b.dependency("zli", .{
        .target = target,
        .optimize = optimize,
    });
    const zli_module = zli_dependency.module("zli");

    const mod = b.addModule("zjpeg2png", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigimg", .module = zigimg_module },
        },
    });

    const exe = b.addExecutable(.{
        .name = "zjpeg2png",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zjpeg2png", .module = mod },
                .{ .name = "zigimg", .module = zigimg_module },
                .{ .name = "zli", .module = zli_module },
            },
        }),
        .use_llvm = true, // zigimg needs
    });
    exe.root_module.addOptions("zon", options);
    if (optimize == .ReleaseFast) exe.root_module.strip = true;

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
        .use_llvm = true, // zigimg needs
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
        .use_llvm = true, // zigimg needs
    });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // Pipeline regression/determinism guard: decodes tests/image.jpg (embedded)
    // and pins a SHA-256 of the rendered RGB. Lives in its own module so the
    // @embedFile resolves next to the fixture.
    const regression_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/regression.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zjpeg2png", .module = mod },
            },
        }),
        .use_llvm = true, // zigimg needs
    });
    const run_regression_tests = b.addRunArtifact(regression_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_regression_tests.step);
}
