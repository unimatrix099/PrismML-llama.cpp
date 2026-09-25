// TDD harness for a NEON ggml_vec_dot_ptq1_0_q8_0 on aarch64.
// Reference kernel is copied verbatim from ggml-cpu/quants.c (generic).
// We compare a NEON implementation against it for bit-exactness, then microbench.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define QK_PTQ1_0 128
#define QK8_0 32
#define GGML_RESTRICT __restrict

typedef uint16_t ggml_half;

typedef struct {
    uint8_t qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 24 B
    uint8_t qh[QK_PTQ1_0/64];                   //  2 B
    ggml_half d;
} block_ptq1_0;

typedef struct {
    ggml_half d;
    int8_t qs[QK8_0];
} block_q8_0;

// minimal fp16 <-> fp32 (round-to-nearest-even not needed for test scales)
static inline float fp16_to_fp32(ggml_half h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}
static inline ggml_half fp32_to_fp16(float x) {
    uint32_t f; memcpy(&f, &x, 4);
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 0x1F) return (ggml_half)(sign | 0x7C00);
    return (ggml_half)(sign | (exp << 10) | (mant >> 13));
}
#define GGML_CPU_FP16_TO_FP32(x) fp16_to_fp32(x)

// ---------------- reference (from quants.c) ----------------
void ref_ptq1(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int qk = QK_PTQ1_0;
    const int nb = n / qk;
    const block_ptq1_0 * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    static const size_t  stages[3] = {32, 16, 8};
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        int8_t q[QK_PTQ1_0];
        int o = 0;
        size_t j = 0;
        for (size_t st = 0; st < 3; ++st) {
            const size_t c = stages[st];
            for (; j + c <= sizeof(x->qs); j += c) {
                for (size_t nn = 0; nn < 5; ++nn) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t v  = x[i].qs[j + m] * pow3[nn];
                        const int16_t xi = ((uint16_t) v * 3) >> 8;
                        q[o++] = (int8_t) (xi - 1);
                    }
                }
            }
        }
        for (size_t nn = 0; nn < 4; ++nn) {
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v  = x[i].qh[h] * pow3[nn];
                const int16_t xi = ((uint16_t) v * 3) >> 8;
                q[o++] = (int8_t) (xi - 1);
            }
        }
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;
            for (int b = 0; b < 32; ++b) {
                sumi_block += (int) q[k*32 + b] * (int) yb->qs[b];
            }
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    *s = sumf;
}

// ---------------- NEON version (vectorized unpack + SDOT dot) ----------------
// unpack one trit: v = byte*pow3 (u8 wrap); xi = (u16(v)*3)>>8 in {0,1,2}; q = xi-1.
static inline int8x16_t unpack16(uint8x16_t bytes, uint8_t p) {
    uint8x16_t v = vmulq_u8(bytes, vdupq_n_u8(p));
    uint16x8_t lo = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(v)),  3), 8);
    uint16x8_t hi = vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(v)), 3), 8);
    int8x16_t r = vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(lo), vmovn_u16(hi)));
    return vsubq_s8(r, vdupq_n_s8(1));
}
static inline int8x8_t unpack8(uint8x8_t bytes, uint8_t p) {
    uint8x8_t v = vmul_u8(bytes, vdup_n_u8(p));
    uint16x8_t w = vshrq_n_u16(vmulq_n_u16(vmovl_u8(v), 3), 8);
    return vsub_s8(vreinterpret_s8_u8(vmovn_u16(w)), vdup_n_s8(1));
}
// fused dot of one 32-trit group (two int8x16 halves) against one q8_0 block
static inline float ptq1_dot32(int8x16_t a, int8x16_t b, const block_q8_0 * GGML_RESTRICT yb) {
    int32x4_t acc = vdotq_s32(vdupq_n_s32(0), a, vld1q_s8(yb->qs));
    acc = vdotq_s32(acc, b, vld1q_s8(yb->qs + 16));
    return GGML_CPU_FP16_TO_FP32(yb->d) * (float) vaddvq_s32(acc);
}
void neon_ptq1(int n, float * GGML_RESTRICT s, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy) {
    const int qk = QK_PTQ1_0;
    const int nb = n / qk;
    const block_ptq1_0 * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        __builtin_prefetch((const char *)(x + i) + 512, 0, 0);
        // unpack directly into the 4 dot groups; no memory temp
        const uint8x16_t b16 = vld1q_u8(x[i].qs);       // bytes 0..15 -> trits 0..79
        const uint8x8_t  b8  = vld1_u8(x[i].qs + 16);   // bytes 16..23 -> trits 80..119
        const int8x16_t u0 = unpack16(b16, pow3[0]);
        const int8x16_t u1 = unpack16(b16, pow3[1]);
        const int8x16_t u2 = unpack16(b16, pow3[2]);
        const int8x16_t u3 = unpack16(b16, pow3[3]);
        const int8x16_t u4 = unpack16(b16, pow3[4]);
        const int8x8_t  w0 = unpack8(b8, pow3[0]);
        const int8x8_t  w1 = unpack8(b8, pow3[1]);
        const int8x8_t  w2 = unpack8(b8, pow3[2]);
        const int8x8_t  w3 = unpack8(b8, pow3[3]);
        const int8x8_t  w4 = unpack8(b8, pow3[4]);
        // qh 8 trits -> trits 120..127 (order: qh0,qh1 per nn=0..3), scalar
        int8_t qh8[8]; int o = 0;
        for (int nn = 0; nn < 4; ++nn)
            for (size_t h = 0; h < sizeof(x->qh); ++h) {
                const uint8_t v = x[i].qh[h] * pow3[nn];
                qh8[o++] = (int8_t)((((uint16_t) v * 3) >> 8) - 1);
            }
        const int8x8_t wq = vld1_s8(qh8);
        const block_q8_0 * GGML_RESTRICT yb = &y[i * 4];
        float sumi = ptq1_dot32(u0, u1, yb + 0)
                   + ptq1_dot32(u2, u3, yb + 1)
                   + ptq1_dot32(u4, vcombine_s8(w0, w1), yb + 2)
                   + ptq1_dot32(vcombine_s8(w2, w3), vcombine_s8(w4, wq), yb + 3);
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }
    *s = sumf;
}

// ---------------- test + microbench ----------------
static uint32_t rng = 12345;
static uint32_t xr(void){ rng ^= rng<<13; rng ^= rng>>17; rng ^= rng<<5; return rng; }

int main(void) {
    const int NB = 512;                 // blocks of 128 -> n = NB*128
    const int n  = NB * QK_PTQ1_0;
    block_ptq1_0 * x = malloc(sizeof(block_ptq1_0) * NB);
    block_q8_0   * y = malloc(sizeof(block_q8_0)   * NB * 4);

    int fails = 0;
    for (int trial = 0; trial < 200; ++trial) {
        for (int i = 0; i < NB; i++) {
            for (size_t b = 0; b < sizeof(x[i].qs); b++) x[i].qs[b] = xr() % 243; // valid 5-trit byte
            for (size_t b = 0; b < sizeof(x[i].qh); b++) x[i].qh[b] = xr() % 81;  // valid 4-trit byte
            x[i].d = fp32_to_fp16(((float)(xr()%2000)/1000.0f - 1.0f) * 0.05f);
        }
        for (int i = 0; i < NB*4; i++) {
            for (int b = 0; b < QK8_0; b++) y[i].qs[b] = (int8_t)(xr() % 256);
            y[i].d = fp32_to_fp16(((float)(xr()%2000)/1000.0f) * 0.02f + 0.001f);
        }
        float sref, snew;
        ref_ptq1(n, &sref, x, y);
        neon_ptq1(n, &snew, x, y);
        if (sref != snew) {
            if (fails < 5) printf("MISMATCH trial %d: ref=%.9g neon=%.9g diff=%.3g\n", trial, sref, snew, sref-snew);
            fails++;
        }
    }
    printf(fails ? "FAIL: %d/200 trials mismatched\n" : "PASS: 200/200 trials bit-exact\n", fails);

    // microbench
    const int ITERS = 20000;
    float acc = 0; struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; ++it) { float s; ref_ptq1(n, &s, x, y); acc += s; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tref = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < ITERS; ++it) { float s; neon_ptq1(n, &s, x, y); acc += s; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double tneon = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("microbench (n=%d, %d iters): ref=%.3fs neon=%.3fs speedup=%.2fx  [sink=%.3g]\n",
           n, ITERS, tref, tneon, tref/tneon, acc);
    free(x); free(y);
    return fails ? 1 : 0;
}
