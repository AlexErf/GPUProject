#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Original:
// X_T1491[n0, n1, n2, 736 + a : _T2118, _T2119, _T2120, _T2121] = =(X_T1490[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1491[n0, n1, n2, 736 + a : _T2118, _T2119, _T2120, _T2121] = =(X_T1490[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 32, 0 <= 736 + a < 768, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 32 }
// Defracted:
// X_T1491[n0, n1, n2, 736 + a : _T2118, _T2119, _T2120, _T2121] = =(X_T1490[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1491   X_T1490  
//        a        32         1         1  
//       n1         7      5376       224  
//       n2         7       768        32  
//      off                 736         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 32, 7, 7 }
// Out stride: { 1, 5376, 768 }
// Input 1 offset: 0
// Input 1 stride: { 1, 224, 32 }
// Tile size: { 32, 1, 7 }
// Contraction output var shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Computed true ops: 3136
// Computed work groups: 7
// Computed inner loops: 1
// Computed shared mem: 896
// Computed out regs: 1024
// Computed mem read: 896
// Computed mem write: 896
// Computed operations: 224
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 1, 1
__kernel void kernel_c108_sdk_508(__global float* restrict  X_T1491, __global const float* restrict  in1)
{
  X_T1491 = (X_T1491 + 736);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[224];
  int n1_gid = get_group_id(0);
  {
    {
      int gbase = (n1_gid * 224);
      int a_n2_n1_tid = (tid % 256);
      int a_n2_n1_cond = (a_n2_n1_tid < 224);
      if (a_n2_n1_cond)
      {
        int gidx = (gbase + a_n2_n1_tid);
        in1_shared[a_n2_n1_tid] = in1[clamp((int)gidx, (int)0, (int)1567)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    int n2_cond = (n2_tid < 7);
    int n2 = select((int)0, (int)n2_tid, (int)n2_cond);
    float val1 = in1_shared[(a_tid + (32 * n2))];
    agg[0] = select((float)agg[0], (float)val1, (int)n2_cond);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  int n2_cond = (n2_tid < 7);
  if (n2_cond)
  {
    float LX_T1491 = agg[0];
    int gout_idx = ((a_tid + (5376 * n1_gid)) + (768 * n2_tid));
    X_T1491[gout_idx] = LX_T1491;
  }
}
