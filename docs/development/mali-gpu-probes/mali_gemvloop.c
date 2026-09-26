// Light GPU load: run the realistic 2-bit GEMV in a loop for N seconds.
// Tests whether a USEFUL co-processor load (~2.5 GB/s) contends with CPU decode.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define CK(x) do{cl_int e=(x);if(e!=CL_SUCCESS){printf("CL err %d @%d\n",e,__LINE__);exit(1);}}while(0)
const char* SRC=
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"__kernel void gemv3(__global const uchar16* w, __global const half* x,\n"
"                    __global const half* sc, __global float* y,const int Kb,const int K){\n"
"  int row=get_group_id(0),lid=get_local_id(0),lsz=get_local_size(0);\n"
"  __local char xq0[1024],xq1[1024],xq2[1024],xq3[1024]; int Kq=K/4;\n"
"  for(int k=lid;k<Kq;k+=lsz){int o=k*4;\n"
"   xq0[k]=(char)clamp((int)(vload_half(o,x)*8.f),-127,127);xq1[k]=(char)clamp((int)(vload_half(o+1,x)*8.f),-127,127);\n"
"   xq2[k]=(char)clamp((int)(vload_half(o+2,x)*8.f),-127,127);xq3[k]=(char)clamp((int)(vload_half(o+3,x)*8.f),-127,127);}\n"
"  barrier(CLK_LOCAL_MEM_FENCE);\n"
"  __global const uchar16* wr=w+(long)row*(Kb/16); int Kv=Kb/16,acc=0;\n"
"  for(int c=lid;c<Kv;c+=lsz){uchar16 b=wr[c];\n"
"   char16 q0=convert_char16(b&(uchar16)3)-(char16)1,q1=convert_char16((b>>(uchar16)2)&(uchar16)3)-(char16)1;\n"
"   char16 q2=convert_char16((b>>(uchar16)4)&(uchar16)3)-(char16)1,q3=convert_char16((b>>(uchar16)6)&(uchar16)3)-(char16)1;\n"
"   char16 a0=vload16(c,xq0),a1=vload16(c,xq1),a2=vload16(c,xq2),a3=vload16(c,xq3);\n"
"   int16 pp=convert_int16(q0)*convert_int16(a0)+convert_int16(q1)*convert_int16(a1)+convert_int16(q2)*convert_int16(a2)+convert_int16(q3)*convert_int16(a3);\n"
"   acc+=pp.s0+pp.s1+pp.s2+pp.s3+pp.s4+pp.s5+pp.s6+pp.s7+pp.s8+pp.s9+pp.sa+pp.sb+pp.sc+pp.sd+pp.se+pp.sf;}\n"
"  __local int r[256];r[lid]=acc;barrier(CLK_LOCAL_MEM_FENCE);\n"
"  for(int o=lsz/2;o>0;o>>=1){if(lid<o)r[lid]+=r[lid+o];barrier(CLK_LOCAL_MEM_FENCE);}\n"
"  if(lid==0)y[row]=(float)r[0]*vload_half(row,sc)*(1.f/8.f);}\n";
int main(int argc,char**argv){
 double dur=argc>1?atof(argv[1]):30.0;
 cl_platform_id p[4];cl_uint np;CK(clGetPlatformIDs(4,p,&np));cl_device_id d=0;
 for(cl_uint i=0;i<np;i++){cl_uint nd;if(clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS&&nd)break;}
 cl_int e;cl_context c=clCreateContext(0,1,&d,0,0,&e);CK(e);
 cl_command_queue q=clCreateCommandQueue(c,d,0,&e);CK(e);
 cl_program pr=clCreateProgramWithSource(c,1,&SRC,0,&e);CK(e);CK(clBuildProgram(pr,1,&d,"-cl-fast-relaxed-math",0,0));
 int K=4096,Kb=K/4,M=200000; size_t wb=(size_t)M*Kb;
 cl_mem w=clCreateBuffer(c,CL_MEM_READ_ONLY,wb,0,&e);CK(e);cl_mem x=clCreateBuffer(c,CL_MEM_READ_ONLY,K*2,0,&e);CK(e);
 cl_mem sc=clCreateBuffer(c,CL_MEM_READ_ONLY,(size_t)M*2,0,&e);CK(e);cl_mem y=clCreateBuffer(c,CL_MEM_WRITE_ONLY,(size_t)M*4,0,&e);CK(e);
 {char*z=calloc(1,1<<20);for(size_t o=0;o<wb;o+=(1<<20)){size_t cc=(wb-o<(1<<20))?(wb-o):(1<<20);CK(clEnqueueWriteBuffer(q,w,CL_FALSE,o,cc,z,0,0,0));}clFinish(q);free(z);}
 cl_kernel kg=clCreateKernel(pr,"gemv3",&e);CK(e);
 CK(clSetKernelArg(kg,0,sizeof(cl_mem),&w));CK(clSetKernelArg(kg,1,sizeof(cl_mem),&x));CK(clSetKernelArg(kg,2,sizeof(cl_mem),&sc));
 CK(clSetKernelArg(kg,3,sizeof(cl_mem),&y));CK(clSetKernelArg(kg,4,sizeof(int),&Kb));CK(clSetKernelArg(kg,5,sizeof(int),&K));
 size_t lg=64,gg=(size_t)M*lg; double t0=now(),moved=0;long reps=0;
 while(now()-t0<dur){CK(clEnqueueNDRangeKernel(q,kg,1,0,&gg,&lg,0,0,0));clFinish(q);moved+=(double)wb;reps++;}
 double dt=now()-t0;
 printf("GPU 2-bit GEMV loop: %.2f GB/s bytes, %.1f Gw/s useful, over %.1fs (%ld reps)\n",moved/1e9/dt,(double)M*K*reps/dt/1e9,dt,reps);
 return 0;}
