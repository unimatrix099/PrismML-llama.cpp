# Mali-G610 GPU feasibility probes (RK3588)

Standalone OpenCL microbenchmarks used to decide whether a from-scratch GPU decode path
for Bonsai 2 could beat the CPU NEON kernels. Conclusion: **no-go** — see
`../opencl-mali-experiment.md` for the full analysis and measured numbers.

## Build & run (on the board)
```
gcc -O2 mali_bw.c    -o mali_bw    -lOpenCL && ./mali_bw
gcc -O2 mali_bw2.c   -o mali_bw2   -lOpenCL && ./mali_bw2
gcc -O2 mali_gemv2.c -o mali_gemv2 -lOpenCL && ./mali_gemv2
gcc -O2 mali_gemv3.c -o mali_gemv3 -lOpenCL && ./mali_gemv3
```
Needs `libOpenCL` + `/usr/include/CL` (libmali-g13p0 stack on the Orange Pi 5 Ultra).

## What each measures
- **mali_bw.c**  — first pass: coalesced read bandwidth + a naive 2-bit GEMV.
- **mali_bw2.c** — tuned peak read BW sweep (uint16, wg configs) + int8 GEMV (the
  memory-bound ceiling, ~22.6 Gw/s).
- **mali_gemv2.c** — 2-bit GEMV with activation preloaded to local + uchar16 reads
  (scalar byte unpack — spills, ~5.9 Gw/s).
- **mali_gemv3.c** — fully vectorized 2-bit GEMV: SIMD mask/shift unpack + deinterleaved
  activations (~9.8 Gw/s). Best low-bit result, still 3.7x below the CPU.

Reference: CPU NEON PQ2_0 = 36.3 Gw/s = 1.35 tok/s on the 27B model.
