// v2: find TRUE peak Mali read BW (sweep vec width + wg) + better int8-dot GEMV.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e!=CL_SUCCESS){printf("CL err %d @%d\n",e,__LINE__);exit(1);}}while(0)
const char* SRC=
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void bw16(__global const uint16* in,__global uint* out,const int n,const int iters){\n"
" int gid=get_global_id(0),gsz=get_global_size(0); uint16 a=(uint16)(0);\n"
" for(int it=0;it<iters;it++) for(int i=gid;i<n;i+=gsz) a+=in[i];\n"
" uint s=a.s0+a.s1+a.s2+a.s3+a.s4+a.s5+a.s6+a.s7+a.s8+a.s9+a.sa+a.sb+a.sc+a.sd+a.se+a.sf;\n"
" if(s==0xdeadbeef) out[gid]=1;}\n"
// int8-dot GEMV: weights pre-unpacked to int8 [-1,0,1], activation int8; use dot() over char4
"__kernel void gemv_i8(__global const char4* w,__global const char4* x,__global const half* sc,\n"
"                      __global float* y,const int K4){\n"
" int row=get_group_id(0),lid=get_local_id(0),lsz=get_local_size(0);\n"
" __global const char4* wr=w+(long)row*K4;\n"
" int acc=0;\n"
" for(int c=lid;c<K4;c+=lsz){ char4 wv=wr[c]; char4 xv=x[c]; acc+=wv.x*xv.x+wv.y*xv.y+wv.z*xv.z+wv.w*xv.w; }\n"
" __local int r[256]; r[lid]=acc; barrier(CLK_LOCAL_MEM_FENCE);\n"
" for(int o=lsz/2;o>0;o>>=1){if(lid<o)r[lid]+=r[lid+o];barrier(CLK_LOCAL_MEM_FENCE);}\n"
" if(lid==0) y[row]=(float)r[0]*vload_half(row,sc);}\n";
int main(int argc,char**argv){
 cl_platform_id p[4];cl_uint np;CK(clGetPlatformIDs(4,p,&np));cl_device_id d=0;
 for(cl_uint i=0;i<np;i++){cl_uint nd;if(clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS&&nd)break;}
 cl_int e;cl_context c=clCreateContext(0,1,&d,0,0,&e);CK(e);
 cl_command_queue q=clCreateCommandQueue(c,d,0,&e);CK(e);
 cl_program pr=clCreateProgramWithSource(c,1,&SRC,0,&e);CK(e);
 e=clBuildProgram(pr,1,&d,"-cl-fast-relaxed-math",0,0);
 if(e){size_t ls;clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,0,0,&ls);char*l=malloc(ls+1);clGetProgramBuildInfo(pr,d,CL_PROGRAM_BUILD_LOG,ls+1,l,0);printf("%s\n",l);exit(1);}
 size_t bytes=(size_t)1536*1024*1024; cl_mem in=clCreateBuffer(c,CL_MEM_READ_ONLY,bytes,0,&e);CK(e);
 {char*z=calloc(1,1<<20);for(size_t o=0;o<bytes;o+=(1<<20))CK(clEnqueueWriteBuffer(q,in,CL_FALSE,o,(1<<20),z,0,0,0));clFinish(q);free(z);}
 cl_mem out=clCreateBuffer(c,CL_MEM_WRITE_ONLY,1<<22,0,&e);CK(e);
 cl_kernel k=clCreateKernel(pr,"bw16",&e);CK(e);
 int n16=bytes/64,iters=3;
 CK(clSetKernelArg(k,0,sizeof(cl_mem),&in));CK(clSetKernelArg(k,1,sizeof(cl_mem),&out));
 CK(clSetKernelArg(k,2,sizeof(int),&n16));CK(clSetKernelArg(k,3,sizeof(int),&iters));
 int cfgs[][2]={{16384,64},{32768,128},{65536,256},{131072,256},{262144,256},{65536,128}};
 double best=0;
 for(int i=0;i<6;i++){size_t g=cfgs[i][0],l=cfgs[i][1];
   CK(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0));clFinish(q);
   double t=now();int R=3;for(int r=0;r<R;r++)CK(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0));clFinish(q);double dt=now()-t;
   double gb=(double)bytes*iters*R/1e9,bw=gb/dt; if(bw>best)best=bw;
   printf("BW uint16 g=%zu l=%zu: %.1f GB/s\n",g,l,bw);}
 printf("PEAK read BW: %.1f GB/s  -> ceiling @7.26GB/tok: %.2f tok/s\n",best,best/7.26);
 // int8 GEMV
 int K=4096,K4=K/4,M=120000; size_t wb=(size_t)M*K; // 1 byte/weight (unpacked int8)
 cl_mem w=clCreateBuffer(c,CL_MEM_READ_ONLY,wb,0,&e);CK(e);
 cl_mem x=clCreateBuffer(c,CL_MEM_READ_ONLY,K,0,&e);CK(e);
 cl_mem sc=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)M*2,0,&e);CK(e);
 cl_mem y=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*4,0,&e);CK(e);
 {char*z=calloc(1,1<<20);for(size_t o=0;o<wb;o+=(1<<20)){size_t cc=(wb-o<(1<<20))?(wb-o):(1<<20);CK(clEnqueueWriteBuffer(q,w,CL_FALSE,o,cc,z,0,0,0));}clFinish(q);free(z);}
 cl_kernel kg=clCreateKernel(pr,"gemv_i8",&e);CK(e);
 CK(clSetKernelArg(kg,0,sizeof(cl_mem),&w));CK(clSetKernelArg(kg,1,sizeof(cl_mem),&x));
 CK(clSetKernelArg(kg,2,sizeof(cl_mem),&sc));CK(clSetKernelArg(kg,3,sizeof(cl_mem),&y));CK(clSetKernelArg(kg,4,sizeof(int),&K4));
 size_t lg=64,gg=(size_t)M*lg;
 CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));clFinish(q);
 double t=now();int R=3;for(int r=0;r<R;r++)CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));clFinish(q);double dt=now()-t;
 double gb=(double)wb*R/1e9; // note: int8 here is 8bpw; PQ2 is 2bpw so 4x less traffic
 printf("GEMV int8 (8bpw): %.1f GB/s eff, %.3fs. Equiv PQ2(2bpw) traffic=%.2fGB/tok -> %.2f tok/s\n",
        gb/dt,dt,7.26,(gb/dt)/ (7.26*4));
 printf("  (int8 GEMV BW %.1f GB/s vs peak %.1f = %.0f%% of peak)\n",gb/dt,best,100*(gb/dt)/best);
 return 0;}
