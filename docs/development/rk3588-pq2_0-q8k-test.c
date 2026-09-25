// TDD harness for PQ2_0 x Q8_K vec_dot on aarch64 (the actual hot path).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define QK_PQ2_0 128
#define QK_K 256
#define GGML_RESTRICT __restrict
typedef uint16_t ggml_half;
typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0/4]; } block_pq2_0;       // 32 bytes
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

static inline float f16(ggml_half h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;if(!e){if(!m)f=s;else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|(e<<23)|(m<<13);}}else if(e==0x1F)f=s|0x7F800000|(m<<13);else f=s|((e-15+127)<<23)|(m<<13);float o;memcpy(&o,&f,4);return o;}
static inline ggml_half f32h(float x){uint32_t f;memcpy(&f,&x,4);uint32_t sign=(f>>16)&0x8000;int32_t e=((f>>23)&0xFF)-127+15;uint32_t m=f&0x7FFFFF;if(e<=0)return (ggml_half)sign;if(e>=0x1F)return (ggml_half)(sign|0x7C00);return (ggml_half)(sign|(e<<10)|(m>>13));}
#define GGML_CPU_FP16_TO_FP32(x) f16(x)

// ---- reference (generic) ----
void ref_pq2k(int n, float * s, const void * vx, const void * vy){
    const block_pq2_0*x=vx; const block_q8_K*y=vy; const int nb=n/QK_PQ2_0; float sumf=0;
    for(int i=0;i<nb;i++){ const block_q8_K*yb=&y[i>>1]; const int8_t*q8=yb->qs+128*(i&1); int sumi=0;
        for(int k=0;k<4;k++) for(int b=0;b<8;b++){ uint8_t by=x[i].qs[8*k+b];
            for(int j=0;j<4;j++) sumi += (((by>>(2*j))&3)-1)*q8[32*k+4*b+j];
        }
        sumf += (f16(x[i].d)*yb->d)*(float)sumi;
    }
    *s=sumf;
}

// ---- NEON candidate ----
void neon_pq2k(int n, float * s, const void * vx, const void * vy){
    const block_pq2_0*x=vx; const block_q8_K*y=vy; const int nb=n/QK_PQ2_0; float sumf=0;
    static const uint8_t tbl_idx_lo[16]={0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3};
    static const uint8_t tbl_idx_hi[16]={4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7};
    static const int8_t shift_vals[16]={0,-2,-4,-6,0,-2,-4,-6,0,-2,-4,-6,0,-2,-4,-6};
    const uint8x16_t idx_lo=vld1q_u8(tbl_idx_lo), idx_hi=vld1q_u8(tbl_idx_hi);
    const int8x16_t shifts=vld1q_s8(shift_vals); const uint8x16_t mask2=vdupq_n_u8(0x03); const int8x16_t one=vdupq_n_s8(1);
    for(int i=0;i<nb;i++){ const block_q8_K*yb=&y[i>>1]; const int8_t*q8=yb->qs+128*(i&1);
        int32x4_t acc=vdupq_n_s32(0);
        for(int k=0;k<4;k++){
            const uint8x8_t raw=vld1_u8(&x[i].qs[k*8]); const uint8x16_t raw16=vcombine_u8(raw,raw);
            uint8x16_t b0=vqtbl1q_u8(raw16,idx_lo); int8x16_t qv0=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(b0,shifts),mask2)),one);
            uint8x16_t b1=vqtbl1q_u8(raw16,idx_hi); int8x16_t qv1=vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshlq_u8(b1,shifts),mask2)),one);
            acc=vdotq_s32(acc,qv0,vld1q_s8(q8+32*k));
            acc=vdotq_s32(acc,qv1,vld1q_s8(q8+32*k+16));
        }
        sumf += (f16(x[i].d)*yb->d)*(float)vaddvq_s32(acc);
    }
    *s=sumf;
}

static uint32_t R=777; static uint32_t xr(){R^=R<<13;R^=R>>17;R^=R<<5;return R;}
int main(){
    const int NB=512, n=NB*QK_PQ2_0;   // NB pq2 blocks, NB/2 q8_K blocks
    block_pq2_0*x=malloc(sizeof(block_pq2_0)*NB); block_q8_K*y=malloc(sizeof(block_q8_K)*(NB/2+1));
    int fails=0;
    for(int t=0;t<200;t++){
        for(int i=0;i<NB;i++){ for(int b=0;b<32;b++)x[i].qs[b]=xr()&0xFF; x[i].d=f32h(((float)(xr()%2000)/1000.f-1.f)*0.05f); }
        for(int i=0;i<NB/2;i++){ for(int b=0;b<QK_K;b++)y[i].qs[b]=(int8_t)(xr()&0xFF); y[i].d=((float)(xr()%2000)/1000.f)*0.02f+0.001f; }
        float a,b; ref_pq2k(n,&a,x,y); neon_pq2k(n,&b,x,y);
        if(fabsf(a-b) > 1e-2f*(fabsf(a)+1e-3f)){ if(fails<5)printf("MISMATCH t%d ref=%.6g neon=%.6g\n",t,a,b); fails++; }
    }
    printf(fails?"FAIL: %d/200\n":"PASS: 200/200 within tol\n",fails);
    const int IT=20000; float acc=0; struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0); for(int it=0;it<IT;it++){float v;ref_pq2k(n,&v,x,y);acc+=v;} clock_gettime(CLOCK_MONOTONIC,&t1);
    double tr=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    clock_gettime(CLOCK_MONOTONIC,&t0); for(int it=0;it<IT;it++){float v;neon_pq2k(n,&v,x,y);acc+=v;} clock_gettime(CLOCK_MONOTONIC,&t1);
    double tn=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("microbench: ref(scalar)=%.3fs neon=%.3fs  speedup=%.2fx  [sink=%.3g]\n",tr,tn,tr/tn,acc);
    return fails?1:0;
}
