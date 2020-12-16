#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T1[a : _T0] = =(X_I_0[a])
// With Index Variables Made Integral:
// X_T1[a : _T0] = =(X_I_0[a]), 500000000 + a < 1000000000
// Constraints:{ 0 <= a < 128, 0 <= a < 512, 0 <= 500000000 + a < 1000000000 }
// Merged Parallel Constraints:{ 0 <= a < 128 }
// Defracted:
// X_T1[a : _T0] = =(X_I_0[a]), 500000000 + a < 1000000000
// Flattened:
//              Range      X_T1     X_I_0  
//        a       128         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a }
// Ranges: { 128 }
// Out stride: { 1 }
// Input 1 offset: 0
// Input 1 stride: { 1 }
// Tile size: { 128 }
// Contraction output var shape: fp32(512):(1):2 KiB
// Computed true ops: 256
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 512
// Computed out regs: 1024
// Computed mem read: 512
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c3_sdk_0(__global float* restrict  X_T1, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[128];
  {
    {
      int a_tid = (tid % 128);
      in1_shared[a_tid] = in1[clamp((int)a_tid, (int)0, (int)127)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    float val1 = in1_shared[a_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  float LX_T1 = agg[0];
  X_T1[a_tid] = LX_T1;
}
