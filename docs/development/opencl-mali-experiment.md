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

## Current blocker (gate 6): runtime kernel-launch failure

`clEnqueueNDRangeKernel` fails (`CL_CHECK` abort at ggml-opencl.cpp:1049). The Intel
work-group / local-size parameters are not valid for Mali's execution limits — this is
**per-kernel work-group tuning**, i.e. open-ended porting, not one more patch. Correctness
of the subgroup ops (which assume exactly 16 lanes) would also need per-kernel verification.

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

## Verdict
The tuned CPU NEON kernels (PQ2_0 @ 1.35 t/s, PTQ1_0 @ 0.86 t/s) remain the right path for
Bonsai 2 decode on RK3588. The Mali branch is preserved for reference but should not be
merged or extended.
