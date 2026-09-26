// Continuous GPU memory-read hog: saturates Mali read BW for N seconds.
// Used to test whether concurrent GPU traffic slows CPU decode (bus contention).
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e!=CL_SUCCESS){printf("CL err %d @%d\n",e,__LINE__);exit(1);}}while(0)
const char* SRC=
"__kernel void bw16(__global const uint16* in,__global uint* out,const int n,const int iters){\n"
" int gid=get_global_id(0),gsz=get_global_size(0); uint16 a=(uint16)(0);\n"
" for(int it=0;it<iters;it++) for(int i=gid;i<n;i+=gsz) a+=in[i];\n"
" uint s=a.s0+a.s1+a.s2+a.s3+a.s4+a.s5+a.s6+a.s7+a.s8+a.s9+a.sa+a.sb+a.sc+a.sd+a.se+a.sf;\n"
" if(s==0xdeadbeef) out[gid]=1;}\n";
int main(int argc,char**argv){
 double dur = argc>1?atof(argv[1]):30.0;
 cl_platform_id p[4];cl_uint np;CK(clGetPlatformIDs(4,p,&np));cl_device_id d=0;
 for(cl_uint i=0;i<np;i++){cl_uint nd;if(clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS&&nd)break;}
 cl_int e;cl_context c=clCreateContext(0,1,&d,0,0,&e);CK(e);
 cl_command_queue q=clCreateCommandQueue(c,d,0,&e);CK(e);
 cl_program pr=clCreateProgramWithSource(c,1,&SRC,0,&e);CK(e);
 CK(clBuildProgram(pr,1,&d,"-cl-fast-relaxed-math",0,0));
 size_t bytes=(size_t)768*1024*1024; cl_mem in=clCreateBuffer(c,CL_MEM_READ_ONLY,bytes,0,&e);CK(e);
 {char*z=calloc(1,1<<20);for(size_t o=0;o<bytes;o+=(1<<20))CK(clEnqueueWriteBuffer(q,in,CL_FALSE,o,(1<<20),z,0,0,0));clFinish(q);free(z);}
 cl_mem out=clCreateBuffer(c,CL_MEM_WRITE_ONLY,1<<22,0,&e);CK(e);
 cl_kernel k=clCreateKernel(pr,"bw16",&e);CK(e);
 int n16=bytes/64,iters=2; size_t g=131072,l=256;
 CK(clSetKernelArg(k,0,sizeof(cl_mem),&in));CK(clSetKernelArg(k,1,sizeof(cl_mem),&out));
 CK(clSetKernelArg(k,2,sizeof(int),&n16));CK(clSetKernelArg(k,3,sizeof(int),&iters));
 double t0=now(); double moved=0; long reps=0;
 while(now()-t0<dur){ CK(clEnqueueNDRangeKernel(q,k,1,0,&g,&l,0,0,0)); clFinish(q); moved+=(double)bytes*iters; reps++; }
 double dt=now()-t0;
 printf("GPU hog: %.1f GB/s sustained over %.1fs (%ld reps)\n", moved/1e9/dt, dt, reps);
 return 0;}
