#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ printf("FAIL line %d %s\n",__LINE__,hipGetErrorString(e_)); return 1; } }while(0)
int main(int argc,char**argv){
  const char* cub=argv[1]; int rows=atoi(argv[2]); int cols=atoi(argv[3]);
  int bd=atoi(argv[4]); int iters=atoi(argv[5]);
  long n=(long)rows*cols;
  FILE*f=fopen(cub,"rb"); if(!f){printf("no cubin\n");return 2;}
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  std::vector<char> img(sz);
  if(fread(img.data(),1,sz,f)!=(size_t)sz){printf("read err\n");return 2;} fclose(f);
  CK(hipInit(0));
  hipModule_t mod; CK(hipModuleLoadData(&mod,img.data()));
  hipFunction_t fn; CK(hipModuleGetFunction(&fn,mod,"swiglu_2_INFINI_DTYPE_BF16_1024"));
  void *dout=0,*da=0,*db=0;
  CK(hipMalloc(&dout,n*2)); CK(hipMalloc(&da,n*2)); CK(hipMalloc(&db,n*2));
  CK(hipMemset(dout,0,n*2)); CK(hipMemset(da,0,n*2)); CK(hipMemset(db,0,n*2));
  hipDeviceptr_t po=(hipDeviceptr_t)dout, pa=(hipDeviceptr_t)da, pb=(hipDeviceptr_t)db;
  int r=rows,c=cols,s=cols,one=1;
  void* args[15]={&po,&r,&c,&s,&one,&pa,&r,&c,&s,&one,&pb,&r,&c,&s,&one};
  unsigned int gx=(unsigned int)((n-1)/1024+1);
  for(int i=0;i<10;i++) CK(hipModuleLaunchKernel(fn,gx,1,1,bd,1,1,0,0,args,0));
  CK(hipDeviceSynchronize());
  hipEvent_t t0,t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
  CK(hipEventRecord(t0));
  for(int i=0;i<iters;i++) CK(hipModuleLaunchKernel(fn,gx,1,1,bd,1,1,0,0,args,0));
  CK(hipEventRecord(t1)); CK(hipEventSynchronize(t1));
  float ms=0; CK(hipEventElapsedTime(&ms,t0,t1));
  double per=ms/iters;
  printf("rows=%d cols=%d n=%ld gx=%u bd=%d iters=%d -> %.5f ms  %.1f GB/s\n",
         rows,cols,n,gx,bd,iters,per,(double)n*6.0/(per/1e3)/1e9);
  return 0;
}
