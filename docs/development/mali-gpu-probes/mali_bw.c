// Standalone Mali-G610 OpenCL memory-bandwidth + 2-bit GEMV probe (RK3588).
// Decides whether a from-scratch GPU decode path could beat CPU NEON.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
#define CK(x) do{cl_int e=(x); if(e!=CL_SUCCESS){printf("CL err %d at %s:%d\n",e,__FILE__,__LINE__);exit(1);} }while(0)

const char* SRC =
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
// ---- pure read bandwidth: each thread reads a contiguous run of uint4, accumulates ----
"__kernel void bw_read(__global const uint4* in, __global uint* out, const int per){\n"
"  int gid = get_global_id(0);\n"
"  uint4 acc = (uint4)(0);\n"
"  int base = gid*per;\n"
"  for(int i=0;i<per;i++){ acc += in[base+i]; }\n"
"  if((acc.x+acc.y+acc.z+acc.w)==0xdeadbeef) out[gid]=1;\n"
"}\n"
// ---- coalesced read: consecutive threads read consecutive uint4 (grid-stride) ----
"__kernel void bw_read_coal(__global const uint4* in, __global uint* out, const int n, const int iters){\n"
"  int gid = get_global_id(0); int gsz = get_global_size(0);\n"
"  uint4 acc=(uint4)(0);\n"
"  for(int it=0;it<iters;it++) for(int i=gid;i<n;i+=gsz){ acc += in[i]; }\n"
"  if((acc.x+acc.y+acc.z+acc.w)==0xdeadbeef) out[gid]=1;\n"
"}\n"
// ---- 2-bit GEMV (PQ2_0-like): W is 2-bit packed [M x K], per-256 fp16 scale; y=W.x ----
// one workgroup per output row; subgroup/threads cooperate over K.
"__kernel void gemv_2bit(__global const uchar* qs,   // packed 2-bit, 4 vals/byte\n"
"                        __global const half*  sc,   // one scale per 256 cols\n"
"                        __global const half*  xin,   // activation [K]\n"
"                        __global float* y,          // out [M]\n"
"                        const int K){\n"
"  int row = get_group_id(0);\n"
"  int lid = get_local_id(0); int lsz = get_local_size(0);\n"
"  __global const uchar* wrow = qs + (long)row*(K/4);\n"
"  __global const half*  srow = sc + (long)row*(K/256);\n"
"  float acc = 0.f;\n"
"  for(int c = lid*4; c < K; c += lsz*4){\n"
"    uchar b = wrow[c>>2];\n"
"    float sf = vload_half(c>>8, srow); // scale\n"
"    float x0=vload_half(c+0,xin), x1=vload_half(c+1,xin), x2=vload_half(c+2,xin), x3=vload_half(c+3,xin);\n"
"    int q0=(b&3)-1, q1=((b>>2)&3)-1, q2=((b>>4)&3)-1, q3=((b>>6)&3)-1;\n"
"    acc += sf*(q0*x0+q1*x1+q2*x2+q3*x3);\n"
"  }\n"
"  __local float red[256];\n"
"  red[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int o=lsz/2;o>0;o>>=1){ if(lid<o) red[lid]+=red[lid+o]; barrier(CLK_LOCAL_MEM_FENCE);}\n"
"  if(lid==0) y[row]=red[0];\n"
"}\n";

int main(){
  cl_platform_id plat[4]; cl_uint np; CK(clGetPlatformIDs(4,plat,&np));
  cl_device_id dev=0; for(cl_uint i=0;i<np;i++){ cl_uint nd; if(clGetDeviceIDs(plat[i],CL_DEVICE_TYPE_GPU,1,&dev,&nd)==CL_SUCCESS&&nd)break; }
  char nm[256]; clGetDeviceInfo(dev,CL_DEVICE_NAME,256,nm,0); printf("device: %s\n",nm);
  cl_int e; cl_context ctx=clCreateContext(0,1,&dev,0,0,&e);CK(e);
  cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e);CK(e);
  cl_program pr=clCreateProgramWithSource(ctx,1,&SRC,0,&e);CK(e);
  e=clBuildProgram(pr,1,&dev,"-cl-fast-relaxed-math",0,0);
  if(e!=CL_SUCCESS){ size_t ls; clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,0,&ls); char*l=malloc(ls+1); clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,ls+1,l,0); printf("build err:\n%s\n",l); exit(1);}

  // ---------- BANDWIDTH ----------
  size_t MB = 1024; size_t bytes = MB*1024*1024;
  cl_mem in=clCreateBuffer(ctx,CL_MEM_READ_ONLY,bytes,0,&e);CK(e);
  // touch/init to force allocation
  { char* z=calloc(1,1<<20); for(size_t off=0;off<bytes;off+=(1<<20)) CK(clEnqueueWriteBuffer(q,in,CL_FALSE,off,(1<<20),z,0,0,0)); clFinish(q); free(z);}
  int n_u4 = bytes/16; // number of uint4
  cl_mem out=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,4*1024*1024,0,&e);CK(e);

  cl_kernel kc=clCreateKernel(pr,"bw_read_coal",&e);CK(e);
  int gsz=1024*64; int iters=4;
  CK(clSetKernelArg(kc,0,sizeof(cl_mem),&in));
  CK(clSetKernelArg(kc,1,sizeof(cl_mem),&out));
  CK(clSetKernelArg(kc,2,sizeof(int),&n_u4));
  CK(clSetKernelArg(kc,3,sizeof(int),&iters));
  size_t g=gsz, l=256;
  // warmup
  CK(clEnqueueNDRangeKernel(q,kc,1,0,&g,&l,0,0,0)); clFinish(q);
  double t0=now(); int R=3;
  for(int r=0;r<R;r++) CK(clEnqueueNDRangeKernel(q,kc,1,0,&g,&l,0,0,0));
  clFinish(q); double dt=now()-t0;
  double gb = (double)bytes*iters*R/1e9;
  printf("BW coalesced read: %.1f GB/s (%.3f s, %.1f GB moved)\n", gb/dt, dt, gb);

  // ---------- 2-bit GEMV (streams a big weight matrix once) ----------
  int K=4096; int M=100000;                       // ~ one big layer's worth, sized to ~ a few hundred MB
  size_t wbytes=(size_t)M*(K/4);                  // 2-bit packed
  size_t scount=(size_t)M*(K/256);
  cl_mem qs=clCreateBuffer(ctx,CL_MEM_READ_ONLY,wbytes,0,&e);CK(e);
  cl_mem sc=clCreateBuffer(ctx,CL_MEM_READ_ONLY,scount*2,0,&e);CK(e);
  cl_mem x =clCreateBuffer(ctx,CL_MEM_READ_ONLY,K*2,0,&e);CK(e);
  cl_mem y =clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)M*4,0,&e);CK(e);
  { char*z=calloc(1,1<<20); for(size_t off=0;off<wbytes;off+=(1<<20)){ size_t c=(wbytes-off<(1<<20))?(wbytes-off):(1<<20); CK(clEnqueueWriteBuffer(q,qs,CL_FALSE,off,c,z,0,0,0));} clFinish(q); free(z);}
  cl_kernel kg=clCreateKernel(pr,"gemv_2bit",&e);CK(e);
  CK(clSetKernelArg(kg,0,sizeof(cl_mem),&qs));
  CK(clSetKernelArg(kg,1,sizeof(cl_mem),&sc));
  CK(clSetKernelArg(kg,2,sizeof(cl_mem),&x));
  CK(clSetKernelArg(kg,3,sizeof(cl_mem),&y));
  CK(clSetKernelArg(kg,4,sizeof(int),&K));
  size_t lg=64, gg=(size_t)M*lg;
  CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0)); clFinish(q);
  double t1=now(); int RR=3;
  for(int r=0;r<RR;r++) CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));
  clFinish(q); double dt1=now()-t1;
  double wgb=(double)wbytes*RR/1e9;
  printf("GEMV 2-bit: %.1f GB/s eff weight read (%.3f s, %.2f GB weights, M=%d K=%d)\n", wgb/dt1, dt1, wgb, M, K);
  // Project decode t/s for a 27B model @2.16bpw = 7.26 GB weights/token
  printf("  -> projected @7.26GB/token: %.2f tok/s (GEMV BW) | %.2f tok/s (pure BW)\n", (wgb/dt1)/7.26, (gb/dt)/7.26);
  return 0;
}
