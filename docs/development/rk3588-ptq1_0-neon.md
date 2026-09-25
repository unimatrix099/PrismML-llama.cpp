# ARM NEON vec_dot for PTQ1_0 (RK3588 / Orange Pi 5 Ultra)

Native aarch64 NEON kernel for `ggml_vec_dot_ptq1_0_q8_0`, replacing the scalar
generic fallback that ARM used before. Target board: Orange Pi 5 Ultra (Rockchip
RK3588, 4x Cortex-A76 + 4x Cortex-A55, LPDDR5, ARMv8.2 + dotprod, no i8mm/SVE),
CPU-only inference of Bonsai 2 27B (`qwen35`, dense).

## Result

PTQ1_0 token generation (llama-bench, `-ngl 0 -p 0 -n 64 -t 8`, performance
governor, build prism-b10735):

| Build | tg t/s | Note |
|-------|-------:|------|
| generic prebuilt (armv8.2)          | 0.38 | scalar ptq1_0 |
| local native (`GGML_NATIVE=ON`)     | 0.38 | scalar ptq1_0, no gain from `-mcpu` |
| local native + kernel (temp buffer) | 0.67 | 1.76x |
| local native + kernel (fused)       | 0.70 | **1.84x** |

Standalone microbenchmark (see test harness below), one core, cache-resident:
- dot vectorized only (SDOT), unpack still scalar: 1.03x (dot was not the cost)
- unpack + dot vectorized, temp buffer: 2.05x
- unpack + dot fused (no temp): 2.28x

## Why

A `perf` flat profile of PTQ1_0 decode on the board:

```
86.15%  ggml_vec_dot_ptq1_0_q8_0
~11%    threadpool spin (big.LITTLE load imbalance)
 0.10%  ggml_compute_forward_fwht   (Hadamard is NOT the bottleneck)
```

The vec_dot was 86% of decode and had no ARM SIMD path (arch-fallback.h aliased
it to the generic scalar version). The Hadamard rotation, a plausible suspect,
was negligible. Runtime tuning (threads, A76-only affinity) and native `-mcpu`
flags gave no gain; the missing kernel was the whole story.

## The kernel

PTQ1_0 block (`QK_PTQ1_0` = 128 trits in 28 bytes, 1.75 bpw):
- `qs[24]`: 5 trits per byte (base-3 packed)
- `qh[2]` : 4 trits per byte
- `d`     : fp16 scale
It is dotted against 4x `block_q8_0` (4 x 32 = 128), each with its own scale.

Per trit the decode is `v = byte * pow3[nn]` (u8, wraps mod 256), then
`xi = (u16(v) * 3) >> 8` in `{0,1,2}`, then `q = xi - 1` in `{-1,0,1}`.

The NEON version does two things the scalar loop did element by element:
1. Unpack: `vmulq_u8` (the wrapping multiply), widen with `vmovl_u8`, `x3` and
   `>>8` with `vmulq_n_u16` / `vshrq_n_u16`, then subtract 1. The qh tail
   (8 values) stays scalar.
2. Dot: `ggml_vdotq_s32` (SDOT when available, widening fallback otherwise).

The unpack outputs feed the four 32-trit dot groups directly (`ptq1_dot32`), with
no intermediate `int8 q[128]` buffer; this removes the per-block store+reload and
its store-to-load latency, which was part of the backend stall (see below).

The float accumulation order is kept identical to the generic kernel
(`sumi += d1 * sumi_block; sumf += d0 * sumi`), so the result is bit-exact with
the reference, not just close.

## What did not help, and the ceiling

A `perf stat` of the optimized decode: IPC 1.77, backend cycles idle 29.5%,
frontend idle ~0, cache-miss rate 0.42%. So the kernel is bound by execution and
dependency latency, not instruction fetch or cache misses.

- Software prefetch of the streamed weights: no measurable gain (the hardware
  prefetcher already covers the sequential read; decode is not memory-latency
  bound). The `__builtin_prefetch` is kept as it is harmless and marginally helps
  some runs.
- `-mcpu=native` / native build flags: no gain over the generic armv8.2 build.
- Thread/affinity tuning: `-t 8` (all cores) is already best; A76-only (`-t 4`,
  taskset) is slower. ggml already does dynamic chunk work-stealing, so
  big.LITTLE imbalance is handled; the residual ~9% is per-op barrier sync across
  the ~400 ops per token.
- Speculative decoding: no Bonsai 2 drafter exists, and on a compute-bound CPU
  verifying K draft tokens costs about Kx the compute, so it is a poor fit here.

Decode stays compute-bound on the ternary unpack. The memory-bandwidth roofline
(about 3.8 t/s = 22 GB/s sustained / 5.9 GB) assumes the compute is free, so it is
not reachable on this CPU for this format; pre-unpacking the weights to int8 to
remove the per-token unpack would need about 24 GB (the board has 16 GB). The
largest untried pure-kernel lever is a PTQ1_0 4-row repack GEMV (more independent
dot products in flight); it is a much bigger change with uncertain on-device
payoff given the per-op barrier floor.

## Files

- `ggml/src/ggml-cpu/arch/arm/quants.c` - native `ggml_vec_dot_ptq1_0_q8_0`,
  guarded by `#if defined(__ARM_NEON)`, else calls the generic.
- `ggml/src/ggml-cpu/arch-fallback.h` - dropped the aarch64 alias that routed
  PTQ1_0 to the generic.

## Test / reproduce

`docs/development/rk3588-ptq1_0-neon-test.c` is a standalone harness (no ggml
dependency). It copies the generic reference verbatim, runs the NEON kernel over
200 random block sets, checks bit-exactness, and microbenchmarks both.

```sh
gcc -O3 -march=armv8.2-a+dotprod -o ptq1_test docs/development/rk3588-ptq1_0-neon-test.c -lm
./ptq1_test
# PASS: 200/200 trials bit-exact
# microbench ... speedup ~2.0x
```

On-board A/B (build baseline first, then apply the kernel and rebuild):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j8 --target llama-bench
LD_LIBRARY_PATH=build/bin build/bin/llama-bench \
  -m Ternary-Bonsai-2-27B-PTQ1_0.gguf -ngl 0 -p 0 -n 64 -t 8
```

Note: use `-p 0` to measure generation only; prompt processing on this model is
also ~0.5 t/s, so a full pp512 sweep takes tens of minutes.

## Scope and limits

- PTQ1_0 x Q8_0 only. PQ2_0 already had a native ARM kernel; PQ2_0 x Q8_K and the
  PTQ1_0 GEMV/GEMM repack paths are still generic on ARM.
- Decode is compute-bound on the ternary unpack; measured tg is still well below
  the memory-bandwidth roofline because that roofline assumes free compute (see the
  ceiling section above).
