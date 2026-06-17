# zjpeg2png

**Silky smooth JPEG decoding in pure Zig — no more artifacts.**

zjpeg2png is a complete Zig port of [jpeg2png](https://github.com/victorvde/jpeg2png) (v1.01). Instead of decoding a JPEG the normal way — which fills the information lost to quantization with blocky, ringing noise — it searches for the *smoothest possible picture that encodes to the exact same JPEG file* and writes that as a PNG. No libpng, no C dependencies — only Zig, [zigimg](https://github.com/zigimg/zigimg), and [zli](https://github.com/xcaeser/zli) for the CLI.

jpeg2png-style decoding works best on pictures that should never have been saved as JPEG: charts, logos, line art, and cartoon-style digital drawings. It gives poor results for photographs and other finely textured images. See the [original project's examples](https://github.com/victorvde/jpeg2png/tree/images) for striking before/after comparisons.

## Building

Requires Zig **0.16.0**.

```sh
zig build -Doptimize=ReleaseFast
./zig-out/bin/zjpeg2png --help
```

## Usage

```sh
zjpeg2png picture.jpg                 # writes picture.png (refuses to overwrite)
zjpeg2png picture.jpg -f              # overwrite if it exists
zjpeg2png picture.jpg -o out.png      # explicit output name (always overwrites)
zjpeg2png a.jpg b.jpg -O smooth/ -t 4 # batch into a directory, up to 4 threads
zjpeg2png picture.jpg -i 1000 -q      # more iterations, no progress bar
zjpeg2png info picture.jpg            # inspect a JPEG without decoding it
zjpeg2png version
```

The CLI is built on [zli](https://github.com/xcaeser/zli): styled `--help`, subcommands, bundled short flags (`-fq`), both `--flag value` and `--flag=value` forms, and an animated progress bar (spinner, smooth sub-character fill, percentage, iteration/file counters, ETA) with per-file ✔ lines in batch mode.

**Subcommands**

- `zjpeg2png <files...>` — the default: smooth-decode JPEGs to PNG
- `zjpeg2png info <file>` — print dimensions, coding (baseline/progressive), chroma subsampling, per-channel block geometry, and quantization tables
- `zjpeg2png version` — version information (also `-V`)

**Decode flags**

| flag | long form | meaning |
|---|---|---|
| `-o file` | `--output` | output file name (single input only; always overwrites) |
| `-O dir` | `--output-dir` | write derived names (`stem.png`) into a directory for batches (created if missing) |
| `-f` | `--force` | overwrite existing derived output names |
| `-w w[,wcb,wcr]` | `--second-order-weight` | TGV second-order weight; higher = smoother transitions, less staircasing; `0` = plain TV (faster). Default `0.3` |
| `-p p[,pcb,pcr]` | `--probability-weight` | DCT coefficient distance weight; higher = closer to the normal JPEG decoding; `0` = off (faster). Default `0.001` |
| `-i n[,ncb,ncr]` | `--iterations` | optimization steps; more = smoother but slower. Default `50` |
| `-q` | `--quiet` | no progress bar |
| `-s` | `--separate-components` | optimize Y/Cb/Cr independently (faster, parallelizes; component edges may disagree) |
| `-t n` | `--threads` | maximum threads; `0` = CPU count (default) |
| `-1` | `--16-bits-png` | 16-bit PNG output (use many iterations with this) |
| `-c file` | `--csv-log` | per-iteration objective values as CSV |
| `-h`, `-V` | `--help`, `--version` | help / version |

Per-channel comma values for `-w`/`-i` require `-s`.

Supported inputs: 8-bit baseline (SOF0) and progressive (SOF2) Huffman JPEGs with 1–4 components — color (YCbCr/RGB), grayscale, and CMYK/YCCK, matching the [let-def fork](https://github.com/let-def/jpeg2png) of jpeg2png.

## How it works

JPEG encoding rounds DCT coefficients to multiples of the quantization step. Standard decoding pretends the rounded values are exact — that mismatch *is* the artifacts. But every image whose coefficients lie within ±½ quantization step of the stored values would have produced the same file, and that set contains much better-looking candidates.

zjpeg2png picks the candidate minimizing

```
TV(u) + w · TGV²(u) + p · Σ (DCT(u − u₀)/quant)²
```

— first-order total variation for smoothness, a second-order term to avoid staircasing, and a small pull toward the standard decoding — using projected subgradient descent with FISTA acceleration. Chroma subsampling is handled by optimizing at full resolution and projecting through the subsampled DCT constraints. The math is explained in detail in the [jpeg2png README](https://github.com/victorvde/jpeg2png#nitty-gritty).

## Performance

1920×1080, 50 iterations, AVX2 machine (16 threads), best of 3:

| variant | joint (default) | separate (`-s`) |
|---|---|---|
| C scalar (no SIMD, 1 thread) | 10.8 s | 6.0 s |
| C SSE2 (1 thread) | 5.9 s | 3.5 s |
| C SSE2 + OpenMP | 5.7 s | 2.0 s |
| **zjpeg2png** | **2.5 s** | **1.4 s** |

How: the TV/TGV/DCT-distance kernels follow the C project's own SSE2 design at AVX2 width (8 × f32 lanes, `@select` zero-norm masking, scatter ordering that preserves scalar accumulation), the 8×8 DCT is row-vectorized with `@shuffle` transposes (the C never vectorized it), hot buffers are 64-byte aligned (`alignedAlloc`), and a single image is optimized on multiple threads (channel-parallel phases plus even/odd row-strip TV/TGV).

Exactness: with `-t 1` the output is **bit-exact** vs the C scalar build — the vector kernels are lane-for-lane identical math. With threads (the default), results are still deterministic, but row-strip seams accumulate in a different order than the serial code: measured against the C reference, ≤0.4% of bytes differ by at most ±1. The auto thread count caps at 8 for a single image (memory-bandwidth bound beyond that); explicit `-t N` is honored as given.

## Testing

```sh
zig build test -Doptimize=ReleaseFast   # recommended
zig build test                          # Debug: all safety checks on
```

## Using as a library

The build exposes a `zjpeg2png` module:

```zig
const zjpeg2png = @import("zjpeg2png");

const log = zjpeg2png.logger.Logger{};
var rendered = try zjpeg2png.pipeline.decodeToPixels(gpa, jpeg_bytes, .{
    .iterations = .{ 100, 100, 100 },
}, null, &log);
defer rendered.deinit(gpa);
// rendered.rgb8: w*h*3 bytes
```

Lower-level pieces (`zjpeg2png.jpeg.readCoefficients`, `zjpeg2png.compute.compute`, `zjpeg2png.dct`, `zjpeg2png.png`) are exported from `src/root.zig`.

## Differences from the C version

Pixel output never differs. The CLI intentionally diverges from jpeg2png's gopt interface: `-o` is no longer repeatable (use `--output-dir` for batches), `info`/`version` subcommands exist, `-h` exits 0, and number parsing is strict where C's `sscanf` ignored trailing garbage. Other deviations: system error messages don't include `strerror` text; CSV float formatting can in principle differ in the last printed digit; restart-marker handling is inherited from zigimg's scan decoder (RST markers are consumed at the bit level; files using restart intervals are decoded on a best-effort basis).

## License

GPL-3.0-or-later, like jpeg2png, from which most of this code is translated. The JPEG entropy decoder in `src/jpeg/` is adapted from [zigimg](https://github.com/zigimg/zigimg) (MIT). The CLI uses [zli](https://github.com/xcaeser/zli) (MIT). The 8×8 DCT is Takuya Ooura's (permissive; notice preserved in `src/dct.zig`).

Credit for the idea, the algorithm, and the original implementation goes to [victorvde/jpeg2png](https://github.com/victorvde/jpeg2png); gray/CMYK handling follows [let-def/jpeg2png](https://github.com/let-def/jpeg2png).

## AI Disclaimer

This code was AI-assisted.