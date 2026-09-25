// Make-or-break: can a TUNED 2-bit (PQ2-like) GEMV on Mali approach peak byte-BW?
// If yes -> from-scratch GPU decode beats CPU. If it stays compute-bound -> parity.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e!=CL_SUCCESS){printf("CL err %d @%d\n",e,__LINE__);exit(1);}}while(0)
const char* SRC=
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
// Tuned 2-bit GEMV. Activation quantized to int8 in local mem once per WG.
// Weights read as uchar16 (64 weights/thread/iter), unpacked, int MAC.
"__kernel void gemv2(__global const uchar16* w, // 2-bit packed, 16B=64 weights\n"
"                    __global const half* x,    // activation [K]\n"
"                    __global const half* sc,   // 1 scale/row (simplified)\n"
"                    __global float* y, const int Kb /*bytes/row=K/4*/, const int K){\n"
"  int row=get_group_id(0),lid=get_local_id(0),lsz=get_local_size(0);\n"
"  __local char xq[8192];\n"                       // up to K=8192 int8 activations
"  for(int i=lid;i<K;i+=lsz){ float v=vload_half(i,x); int q=(int)(v*8.f); xq[i]=(char)clamp(q,-127,127);} \n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  __global const uchar16* wr = w + (long)row*(Kb/16);\n"
"  int acc=0; int Kv=Kb/16;\n"
"  for(int c=lid;c<Kv;c+=lsz){ uchar16 b=wr[c]; int base=c*64;\n"
"    #pragma unroll\n"
"    for(int j=0;j<16;j++){ uchar bb=((uchar*)&b)[j]; int o=base+j*4;\n"
"      acc += ((int)(bb&3)-1)*xq[o] + ((int)((bb>>2)&3)-1)*xq[o+1]\n"
"           + ((int)((bb>>4)&3)-1)*xq[o+2] + ((int)((bb>>6)&3)-1)*xq[o+3]; }\n"
"  }\n"
"  __local int r[256]; r[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int o=lsz/2;o>0;o>>=1){if(lid<o)r[lid]+=r[lid+o];barrier(CLK_LOCAL_MEM_FENCE);}\n"
"  if(lid==0) y[row]=(float)r[0]*vload_half(row,sc)*(1.f/8.f);}\n";
int main(){
 cl_platform_id p[4];cl_uint np;CK(clGetPlatformIDs(4,p,&np));cl_device_id d=0;
 for(cl_uint i=0;i<np;i++){cl_uint nd;if(clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS&&nd)break;}
 cl_int e;cl_context c=clCreateContext(0,1,&d,0,0,&e);CK(e);
 cl_command_queue q=clCreateCommandQueue(c,d,0,&e);CK(e);
 cl_program pr=clCreateProgramWithSource(c,1,&SRC,0,&e);CK(e);
 e=clBuildProgram(pr,1,&d,"-cl-fast-relaxed-math",0,0);
 if(e){size_t ls;clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,0,0,&ls);char*l=malloc(ls+1);clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,ls+1,l,0);printf("%s\n",l);exit(1);}
 int K=4096,Kb=K/4,M=200000; size_t wb=(size_t)M*Kb; // 2bpw packed
 printf("weights=%.2f GB (M=%d K=%d @2bpw)\n",wb/1e9,M,K);
 cl_mem w=clCreateBuffer(c,CL_MEM_READ_ONLY,wb,0,&e);CK(e);
 cl_mem x=clCreateBuffer(c,CL_MEM_READ_ONLY,K*2,0,&e);CK(e);
 cl_mem sc=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)M*2,0,&e);CK(e);
 cl_mem y=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*4,0,&e);CK(e);
 {char*z=calloc(1,1<<20);for(size_t o=0;o<wb;o+=(1<<20)){size_t cc=(wb-o<(1<<20))?(wb-o):(1<<20);CK(clEnqueueWriteBuffer(q,w,CL_FALSE,o,cc,z,0,0,0));}clFinish(q);free(z);}
 cl_kernel kg=clCreateKernel(pr,"gemv2",&e);CK(e);
 CK(clSetKernelArg(kg,0,sizeof(cl_mem),&w));CK(clSetKernelArg(kg,1,sizeof(cl_mem),&x));
 CK(clSetKernelArg(kg,2,sizeof(cl_mem),&sc));CK(clSetKernelArg(kg,3,sizeof(cl_mem),&y));
 CK(clSetKernelArg(kg,4,sizeof(int),&Kb));CK(clSetKernelArg(kg,5,sizeof(int),&K));
 for(int L=64;L<=256;L*=2){ size_t lg=L,gg=(size_t)M*lg;
   CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));clFinish(q);
   double t=now();int R=4;for(int r=0;r<R;r++)CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));clFinish(q);double dt=now()-t;
   double gb=(double)wb*R/1e9,bw=gb/dt;
   printf("gemv2 L=%d: %.1f GB/s bytes | %.2f Gw/s | proj 27B@2bpw(7.26GB): %.2f tok/s\n",
          L,bw,(double)M*K*R/dt/1e9,bw/7.26);}
 return 0;}
