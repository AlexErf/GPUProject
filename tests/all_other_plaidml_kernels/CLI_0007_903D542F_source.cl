#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Original:
// X_T4[256 + a : _T3] = =(X_I_2[a])
// With Index Variables Made Integral:
// X_T4[256 + a : _T3] = =(X_I_2[a]), 500000000 + a < 1000000000
// Constraints:{ 0 <= a < 256, 0 <= 256 + a < 512, 0 <= 500000000 + a < 1000000000 }
// Merged Parallel Constraints:{ 0 <= a < 256 }
// Defracted:
// X_T4[256 + a : _T3] = =(X_I_2[a]), 500000000 + a < 1000000000
// Flattened:
//              Range      X_T4     X_I_2  
//        a       256         1         1  
//      off                 256         0  
//      vec                   1         1  
// 
// Names: { a }
// Ranges: { 256 }
// Out stride: { 1 }
// Input 1 offset: 0
// Input 1 stride: { 1 }
// Tile size: { 256 }
// Contraction output var shape: fp32(512):(1):2 KiB
// Computed true ops: 512
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 1024
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c3_sdk_3(__global float* restrict  X_T4, __global const float* restrict  in1)
{
  X_T4 = (X_T4 + 256);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[256];
  {
    {
      int a_tid = (tid % 256);
      in1_shared[a_tid] = in1[clamp((int)a_tid, (int)0, (int)255)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    float val1 = in1_shared[a_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  float LX_T4 = agg[0];
  X_T4[a_tid] = LX_T4;
}
