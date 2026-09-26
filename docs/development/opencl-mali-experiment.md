# OpenCL / Mali-G610 offload experiment (RK3588)

Branch: `opt/pq2-opencl-mali`
Goal: get the Mali-G610 GPU to run inference via the (Adreno/Intel-focused) `ggml-opencl`
backend, as a step toward GPU offload for Bonsai 2 decode.

## Result: BLOCKED — not a bounded experiment, and no expected payoff

Got the Mali GPU from "unsupported" all the way to executing matmul/norm kernels on
the graph, but the port is nowhere near complete and the physics caps the upside below
the existing CPU path. **Recommendation: do not pursue GPU offload for decode on this SoC.**

## Gates cleared (each was a separate Adreno/Intel assumption)

1. **Device acceptance** — backend hard-rejects non-Adreno/Intel. Added `GPU_FAMILY::MALI`
   + accept branch for `"Mali"` device names.
2. **Build flag** — `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` (Adreno image/bin kernels).
3. **Kernel compile macros** — kernels self-detect `INTEL_GPU`/`ADRENO_GPU` via driver
   extensions (`cl_intel_required_subgroup_size` / `cl_qcom_reqd_sub_group_size`). Mali
   exposes **neither**, so `N_SIMDWIDTH`/`N_DST` and `REQD_SUBGROUP_SIZE_*` were undefined.
   Fixed by host-injecting `-DINTEL_GPU=1` + empty-defining the four `REQD_SUBGROUP_SIZE_*`
   attribute macros (Mali's natural subgroup width is 16 = the Intel path).
4. **Subgroup-size dispatch** — ~5 `sgs`/`sg_size` switches (`rms_norm`, `norm`, `group_norm`,
   `l2_norm`, `gated_delta_net`) asserted on non-Adreno/Intel. Defaulted Mali → 16.
5. **Matmul/op dispatch** — ~39 `if (INTEL){…} else if (ADRENO){…} else ASSERT` sites in
   `ggml_cl_mul_mat` etc. Made the INTEL branch (sg=16) also accept MALI.

## Gate 6: launch VALIDITY — one generic fix (not open-ended)

`clEnqueueNDRangeKernel` fails with **`CL_INVALID_WORK_GROUP_SIZE` (-54)** — first on
`kernel_sigmoid_f32` launched with `gws=[32]` but `lws=[64]` (workgroup larger than the
problem). The Adreno/Intel drivers tolerated invalid geometries; Mali's `libmali` enforces
the spec. Fixed generically by clamping `lws[i] = min(lws[i], gws[i])` (and down to a divisor
of `gws[i]`) in the shared `enqueue_ndrange_kernel` helper. With that ONE change the whole
LFM2.5 Q4_0 graph runs end-to-end on Mali — no crashes.

## Gate 7: launch CORRECTNESS — this is the real, open-ended blocker

The model runs but emits **garbage** (`"The capital of France is ... umps6666666..."`), at
**8.40 t/s — already slower than the same model on CPU (8.72 t/s)**. The kernels were written
against Adreno/Qualcomm and Intel **subgroup semantics**: forcing the Intel path + empty-
defining the required-subgroup-size attribute (Mali ignores it) lets the driver pick a subgroup
width that doesn't match each kernel's reduction/broadcast logic, and the lws clamp further
perturbs the workgroup sizes those reductions depend on. Result: silent numerical corruption
across every subgroup-dependent kernel (matmul, norms, softmax, …). Fixing this is **per-kernel
correctness validation**, i.e. the genuinely open-ended work — and it lands slower than CPU even
if completed. This is why the port is abandoned.

## Why it isn't worth finishing

- **Feasibility of the driver is fine**: Mali-G610 supports `cl_khr_subgroups`
  (shuffle/ballot/non-uniform arithmetic) and `cl_arm_integer_dot_product_accumulate_int8`.
  The blocker is the backend's pervasive Intel-vs-Adreno bifurcation (~60 sites), not the HW.
- **Wrong target**: this only lights up standard quants (Q4_0…). Bonsai 2 (PQ2_0/PTQ1_0)
  has **no** OpenCL kernels — the ternary/2-bit matmul *and* the per-token Hadamard rotation
  would have to be written from scratch after the port is done.
- **Bandwidth ceiling**: decode is memory-bound. The Mali GPU shares the **same LPDDR5**
  (~25 GB/s sustained) as the 8 CPU cores — no separate VRAM, no bandwidth advantage. The
  CPU already runs a tuned NEON PQ2_0 kernel (1.35 t/s). Published RK3588 llama.cpp OpenCL
  results are consistently *slower* than CPU for token generation. Best realistic case here
  is parity, more likely a regression from dispatch overhead + untuned Mali work-group sizes.

## Verdict (port)
The tuned CPU NEON kernels (PQ2_0 @ 1.35 t/s, PTQ1_0 @ 0.86 t/s) remain the right path for
Bonsai 2 decode on RK3588. The Mali branch is preserved for reference but should not be
merged or extended.

---

# From-scratch GPU feasibility study (measure-first, standalone OpenCL probes)

Rather than port the backend, measure whether ANY from-scratch Mali kernel could beat the
CPU. Probes: `scratchpad/mali_bw*.c`, `mali_gemv*.c` (raw C + libOpenCL on the board).

## Device facts (clinfo)
Mali-G610 r0p0 (MP4): **4 compute units @ 1000 MHz**, 32 KB "local" memory whose type is
**`Global`** (no on-chip SRAM scratchpad — tiling does not hide bandwidth), fp16 native,
`cl_khr_subgroups`, `cl_arm_integer_dot_product_accumulate_int8`. Shares the 16 GB LPDDR5
with the CPU (no VRAM).

## Measured throughput (weight-throughput is the deciding unit; token/s = Gw/s ÷ 26.9)
| Path | GB/s | Gw/s | 27B proj t/s |
|------|-----:|-----:|-------------:|
| **CPU NEON PQ2_0 (actual, deployed)** | ~9.8 eff | **36.3** | **1.35** |
| Mali peak coalesced read (uint16, tuned) | **19.7** | — | 2.72 ceiling |
| Mali int8 GEMV (no unpack, memory-bound @115% read-peak) | 22.7 | 22.6 | 0.84 |
| Mali 2-bit GEMV, vectorized unpack + deinterleaved act | 2.5 | 9.8 | 0.37 |
| Mali 2-bit GEMV, scalar unpack | 1.5 | 5.9 | 0.22 |

## Conclusion: NO-GO. A from-scratch GPU decode path cannot beat the CPU here.
- **The GPU's absolute ceiling is below the CPU.** Even the trivial int8 GEMV — no unpack,
  running memory-bound at ~peak read BW — reaches only **22.6 Gw/s**, already below the CPU's
  measured **36.3 Gw/s**. Any real low-bit kernel adds unpack work and is *slower* than that.
- **Mali peak read BW (19.7 GB/s) < the CPU's effective rate for this workload.** The 4×A76
  cores with SDOT/i8mm NEON + a real cache hierarchy simply out-throughput the 4-core iGPU.
- **2-bit unpack is ALU-bound on Mali**, not memory-bound: 2 bpw gives 4× memory headroom
  (2.5 GB/s used vs 19.7 available) but the mask/shift/convert/reduce work caps it at
  ~10 Gw/s. `arm_dot` would help the MAC but not the unpack/convert bottleneck, and cannot
  lift it past the int8 memory-bound ceiling of 22.6 Gw/s.
- **Unified memory removes the usual GPU win.** No VRAM, no fast scratchpad (local=DRAM),
  same LPDDR5 bus — the GPU cannot add bandwidth in parallel with the CPU.

**Net: a from-scratch Mali kernel projects to 0.4–0.8 t/s vs the CPU's 1.35 t/s — a 2–3×
regression — for a large implementation effort (custom PQ2/PTQ1 kernels + repacked weight
layout + Hadamard + attention/KV glue). Not worth building.** The CPU NEON path is not just
convenient, it is genuinely the faster engine for low-bit GEMV decode on this SoC.

Where the board's GPU *could* pay off (not decode): large-batch **prompt/prefill** (compute-
bound GEMM, where the GPU's FLOPS matter and memory is amortized) — but that is not the
1-token/step decode bottleneck the user is optimizing.

---

# NPU + heterogeneous CPU+GPU decode study (2026-09-25)

Follow-up: can the **NPU**, or **CPU+GPU together**, speed up decode?

## NPU (RKNPU / RK3588 6-TOPS): NO-GO on availability AND fundamentals
- Kernel driver *is* loaded (`RKNPU fdab0000.npu` in dmesg), but the **userspace stack is
  absent**: no `librknnrt.so`, no `.rknn` models, no rknn-toolkit / python bindings, and
  **no rknpu backend in the fork's ggml** (grep clean).
- RKNN's op set targets CNN/vision + encoder transformers — **not autoregressive LLM decode**
  (KV cache, per-token Hadamard rotation, dynamic shapes are unsupported).
- **INT8-only**: a 27B model at INT8 = 27 GB > 16 GB RAM (won't fit); no ternary/2-bit path.
- Memory-bound on the same LPDDR5 regardless. Multiple independent blockers → not viable.

## Heterogeneous CPU+GPU: the ONE path with positive headroom (~1.1–1.2× realistic)
Measured shared-bus behaviour (PQ2_0 27B, `-n 32 -t 8`):

| Scenario | CPU decode | GPU useful work | Note |
|----------|-----------:|----------------:|------|
| CPU only (baseline) | **1.33 t/s** | — | uses ~9.8 GB/s of ~25 bus |
| CPU + GPU full memory hog (19 GB/s) | **0.82 t/s** (−38%) | 0 (waste) | proves bus contention; **aggregate ceiling ≈ 25 GB/s** (5.95 CPU + 19 GPU) |
| CPU + GPU running *useful* 2-bit GEMV | **1.30 t/s** (−2%) | **9.9 Gw/s** @ 2.5 GB/s | co-processor coexists almost free |

**Interpretation:** the CPU decode is compute/barrier-bound (~9.8 GB/s of a ~25 GB/s bus), so a
*light* GPU co-processor (2.5 GB/s) runs alongside it at only ~2% cost while adding ~9.9 Gw/s.
Perfectly pipelined that is 35.4 + 9.9 = **45.3 Gw/s ≈ 1.68 t/s (1.26× ceiling)**. A full-blast
GPU instead *robs* the CPU (−38%) — the margin is thin because the bus ceiling is only ~2.5× the
CPU's usage, so the GPU must be bandwidth-throttled to its useful share.

**Why it is still not worth building now:**
- Realistic gain after tensor-parallel sync overhead (split every matmul CPU/GPU, barrier per
  layer, marshal activations + Hadamard both sides, ~420 GPU dispatches/token) is ~**1.1–1.2×**,
  not 1.26×.
- Prerequisites that don't exist: **working Mali OpenCL kernels** (the port above is blocked)
  and **tensor-parallel decode infrastructure** in the fork (absent).
- Contrast the ROI: the CPU NEON kernels already delivered 2.26× (PTQ1_0) and 4.2× (PQ2_0) for
  far less risk. A ~15% heterogeneous gain for a multi-week build is low priority.

## Overall verdict for GPU/NPU decode
NPU: not viable. GPU-alone: 2–3× regression. **CPU+GPU heterogeneous: physically possible with
~1.1–1.2× headroom, but large effort + thin contention margin.** Keep decode on the tuned CPU
NEON path; revisit heterogeneous only if a working Mali GEMV kernel and TP-decode infra already
exist for another reason. Probes: `mali_hog.c`, `mali_gemvloop.c`.
