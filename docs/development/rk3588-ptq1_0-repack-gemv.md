# PTQ1_0 4-row repack GEMV on RK3588 - attempt and negative result

This branch adds a full 4-row repack GEMV path for PTQ1_0 (mirroring the
`block_q1_0x4` / `block_pq2_0` repack machinery): a concatenated `block_ptq1_0x4`
block, `repack_ptq1_0_to_ptq1_0_4_bl`, a NEON `ggml_gemv_ptq1_0_4x8_q8_0` (4 rows
per call, sum(y) amortized across the group), a scalar `ggml_gemm_ptq1_0_4x8_q8_0`
for prefill, template specializations, and registration in
`ggml_repack_get_optimal_repack_type`.

It is correct (bit-exact vs the per-row reference; the model generates coherent
text) but it is SLOWER than the single-row fused kernel on the board, so it is
kept on this branch and NOT merged.

## Measurements (RK3588, PTQ1_0, tg64, -ngl 0 -t 8, performance governor)

| Kernel | tg t/s |
|---|---|
| fused single-row (branch `opt/ptq1_0-arm-neon`) | 0.70 |
| this repack 4-row GEMV | 0.61 - 0.62 |

The isolated, single-core, cache-resident microbench of the 4-row kernel showed
1.24x over the single-row kernel (see `rk3588-ptq1_0-neon.md`). That gain does not
survive real inference:

- Decode is 8-threaded. The repack path parallelizes over output-row groups with
  its own chunking; ggml's normal `mul_mat` uses dynamic 16-row chunk work-stealing
  that balances the big.LITTLE cores (4x A76 + 4x A55) better for these small GEMV
  ops. The single-thread ILP win is lost to worse multi-thread balance.
- Hoisting sum(y) out of the per-group loop (compute once per activation) did not
  change the result (0.62 -> 0.61), so the redundancy was not the cause.
- The repack buffer also doubles resident weight memory during load (original +
  repacked); on the 16 GB board this OOMs at the model's default 262K context and
  needs a smaller `-c` to load at all.

## Takeaway

For this format on this CPU, the single-row fused NEON kernel is faster than a
4-row repack GEMV. The microbench (single core, one group, cache-resident) is not
a reliable predictor of threaded, many-group, cold-weight decode. Production stays
on the fused kernel.
