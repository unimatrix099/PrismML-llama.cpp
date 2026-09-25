// clean multi-row bench: perturb an input byte each iter to defeat hoisting
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#define QK_PTQ1_0 128
#define QK8_0 32
#define GGML_RESTRICT __restrict
typedef uint16_t ggml_half;
typedef struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } block_ptq1_0;
typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
static inline float f16(ggml_half h){uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,f;if(!e){if(!m)f=s;else{e=113;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;f=s|(e<<23)|(m<<13);}}else if(e==0x1F)f=s|0x7F800000|(m<<13);else f=s|((e-15+127)<<23)|(m<<13);float o;memcpy(&o,&f,4);return o;}
#define FP16 f16
static inline int8x16_t up16(uint8x16_t b,uint8_t p){uint8x16_t v=vmulq_u8(b,vdupq_n_u8(p));uint16x8_t lo=vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_low_u8(v)),3),8),hi=vshrq_n_u16(vmulq_n_u16(vmovl_u8(vget_high_u8(v)),3),8);return vreinterpretq_s8_u8(vcombine_u8(vmovn_u16(lo),vmovn_u16(hi)));}
static inline int8x8_t up8(uint8x8_t b,uint8_t p){uint8x8_t v=vmul_u8(b,vdup_n_u8(p));return vreinterpret_s8_u8(vmovn_u16(vshrq_n_u16(vmulq_n_u16(vmovl_u8(v),3),8)));}
static inline float d32(int8x16_t a,int8x16_t b,const block_q8_0*yb,int sy){int32x4_t ac=vdotq_s32(vdupq_n_s32(0),a,vld1q_s8(yb->qs));ac=vdotq_s32(ac,b,vld1q_s8(yb->qs+16));return FP16(yb->d)*(float)(vaddvq_s32(ac)-sy);}
static const uint8_t P[5]={1,3,9,27,81};
static inline float rowdot(const block_ptq1_0*x,const block_q8_0*yb,const int*sy){
    uint8x16_t b16=vld1q_u8(x->qs);uint8x8_t b8=vld1_u8(x->qs+16);
    int8x16_t u0=up16(b16,P[0]),u1=up16(b16,P[1]),u2=up16(b16,P[2]),u3=up16(b16,P[3]),u4=up16(b16,P[4]);
    int8x8_t w0=up8(b8,P[0]),w1=up8(b8,P[1]),w2=up8(b8,P[2]),w3=up8(b8,P[3]),w4=up8(b8,P[4]);
    int8_t q[8];int o=0;for(int nn=0;nn<4;nn++)for(int h=0;h<2;h++){uint8_t v=x->qh[h]*P[nn];q[o++]=(int8_t)(((uint16_t)v*3)>>8);}
    int8x8_t wq=vld1_s8(q);
    return d32(u0,u1,yb,sy[0])+d32(u2,u3,yb+1,sy[1])+d32(u4,vcombine_s8(w0,w1),yb+2,sy[2])+d32(vcombine_s8(w2,w3),vcombine_s8(w4,wq),yb+3,sy[3]);
}
#define NB 256
#define N (NB*QK_PTQ1_0)
static void sumyof(const block_q8_0*yb,int*sy){for(int k=0;k<4;k++)sy[k]=vaddlvq_s8(vld1q_s8(yb[k].qs))+vaddlvq_s8(vld1q_s8(yb[k].qs+16));}
float g1(const block_ptq1_0*r,const block_q8_0*y){float s=0;for(int i=0;i<NB;i++){const block_q8_0*yb=y+i*4;int sy[4];sumyof(yb,sy);s+=FP16(r[i].d)*rowdot(&r[i],yb,sy);}return s;}
void g2(float o[2],const block_ptq1_0*r0,const block_ptq1_0*r1,const block_q8_0*y){float s0=0,s1=0;for(int i=0;i<NB;i++){const block_q8_0*yb=y+i*4;int sy[4];sumyof(yb,sy);s0+=FP16(r0[i].d)*rowdot(&r0[i],yb,sy);s1+=FP16(r1[i].d)*rowdot(&r1[i],yb,sy);}o[0]=s0;o[1]=s1;}
void g4(float o[4],const block_ptq1_0*r0,const block_ptq1_0*r1,const block_ptq1_0*r2,const block_ptq1_0*r3,const block_q8_0*y){float s0=0,s1=0,s2=0,s3=0;for(int i=0;i<NB;i++){const block_q8_0*yb=y+i*4;int sy[4];sumyof(yb,sy);s0+=FP16(r0[i].d)*rowdot(&r0[i],yb,sy);s1+=FP16(r1[i].d)*rowdot(&r1[i],yb,sy);s2+=FP16(r2[i].d)*rowdot(&r2[i],yb,sy);s3+=FP16(r3[i].d)*rowdot(&r3[i],yb,sy);}o[0]=s0;o[1]=s1;o[2]=s2;o[3]=s3;}
static inline int8x16_t up16m(uint8x16_t b,uint8_t p){return vsubq_s8(up16(b,p),vdupq_n_s8(1));}
static inline int8x8_t up8m(uint8x8_t b,uint8_t p){return vsub_s8(up8(b,p),vdup_n_s8(1));}
static inline float d32m(int8x16_t a,int8x16_t b,const block_q8_0*yb){int32x4_t ac=vdotq_s32(vdupq_n_s32(0),a,vld1q_s8(yb->qs));ac=vdotq_s32(ac,b,vld1q_s8(yb->qs+16));return FP16(yb->d)*(float)vaddvq_s32(ac);}
float g1f(const block_ptq1_0*x,const block_q8_0*y){float sf=0;for(int i=0;i<NB;i++){uint8x16_t b16=vld1q_u8(x[i].qs);uint8x8_t b8=vld1_u8(x[i].qs+16);int8x16_t u0=up16m(b16,P[0]),u1=up16m(b16,P[1]),u2=up16m(b16,P[2]),u3=up16m(b16,P[3]),u4=up16m(b16,P[4]);int8x8_t w0=up8m(b8,P[0]),w1=up8m(b8,P[1]),w2=up8m(b8,P[2]),w3=up8m(b8,P[3]),w4=up8m(b8,P[4]);int8_t q[8];int o=0;for(int nn=0;nn<4;nn++)for(int h=0;h<2;h++){uint8_t v=x[i].qh[h]*P[nn];q[o++]=(int8_t)((((uint16_t)v*3)>>8)-1);}int8x8_t wq=vld1_s8(q);const block_q8_0*yb=y+i*4;sf+=FP16(x[i].d)*(d32m(u0,u1,yb)+d32m(u2,u3,yb+1)+d32m(u4,vcombine_s8(w0,w1),yb+2)+d32m(vcombine_s8(w2,w3),vcombine_s8(w4,wq),yb+3));}return sf;}
static uint32_t R=7;static uint32_t xr(){R^=R<<13;R^=R>>17;R^=R<<5;return R;}
int main(){
    block_ptq1_0*rows=malloc(sizeof(block_ptq1_0)*NB*4);block_q8_0*y=malloc(sizeof(block_q8_0)*NB*4);
    for(int b=0;b<NB*4;b++){for(int j=0;j<24;j++)rows[b].qs[j]=xr()%243;rows[b].qh[0]=xr()%81;rows[b].qh[1]=xr()%81;rows[b].d=0x2c00;for(int j=0;j<32;j++)y[b].qs[j]=xr()%256;y[b].d=0x2400;}
    block_ptq1_0*Rr[4]={rows,rows+NB,rows+2*NB,rows+3*NB};
    const int G=6000;volatile float sink=0;struct timespec a,b;
    // single (4 rows worth)
    clock_gettime(CLOCK_MONOTONIC,&a);
    for(int g=0;g<G;g++){rows[g&1023].qs[0]^=1;for(int r=0;r<4;r++)sink+=g1f(Rr[r],y);}
    clock_gettime(CLOCK_MONOTONIC,&b);double t1=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
    // gemv2 (2x per group -> 4 rows)
    clock_gettime(CLOCK_MONOTONIC,&a);
    for(int g=0;g<G;g++){rows[g&1023].qs[0]^=1;float o[2];g2(o,Rr[0],Rr[1],y);sink+=o[0]+o[1];g2(o,Rr[2],Rr[3],y);sink+=o[0]+o[1];}
    clock_gettime(CLOCK_MONOTONIC,&b);double t2=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
    // gemv4
    clock_gettime(CLOCK_MONOTONIC,&a);
    for(int g=0;g<G;g++){rows[g&1023].qs[0]^=1;float o[4];g4(o,Rr[0],Rr[1],Rr[2],Rr[3],y);sink+=o[0]+o[1]+o[2]+o[3];}
    clock_gettime(CLOCK_MONOTONIC,&b);double t4=(b.tv_sec-a.tv_sec)+(b.tv_nsec-a.tv_nsec)/1e9;
    printf("single(fused,prod)=%.3fs  gemv2=%.3fs (%.2fx)  gemv4=%.3fs (%.2fx)  [sink=%.3g]\n",t1,t2,t1/t2,t4,t1/t4,(float)sink);
    return 0;
}
