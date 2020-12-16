#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Original:
// X_T2251[n0, n1, n2, a : _T3233, _T3234, _T3235, _T3236] = =(X_T2250[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2251[n0, n1, n2, a : _T3233, _T3234, _T3235, _T3236] = =(X_T2250[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1472, 0 <= a < 1504, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1472 }
// Defracted:
// X_T2251[n0, n1, n2, a : _T3233, _T3234, _T3235, _T3236] = =(X_T2250[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2251   X_T2250  
//        a      1472         1         1  
//       n1         7     10528     10304  
//       n2         7      1504      1472  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1472, 7, 7 }
// Out stride: { 1, 10528, 1504 }
// Input 1 offset: 0
// Input 1 stride: { 1, 10304, 1472 }
// Tile size: { 128, 1, 7 }
// Contraction output var shape: fp32(1, 7, 7, 1504):(73696, 10528, 1504, 1):287.875 KiB
// Computed true ops: 144256
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 3584
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c124_sdk_780(__global float* restrict  X_T2251, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[896];
  int a_gid = (get_group_id(1) * 128);
  int n1_gid = get_group_id(0);
  {
    {
      int gbase = (a_gid + (n1_gid * 10304));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 3) || (n2_n1_tid < 1));
        if (n2_n1_cond)
        {
          int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
          int lidx = ((7 * a_tid) + n2_n1);
          int gidx = ((gbase + a_tid) + (1472 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)72127)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      int n2_cond = (n2_tid < 7);
      int n2 = select((int)0, (int)n2_tid, (int)n2_cond);
      float val1 = in1_shared[((7 * a) + n2)];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)n2_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 4; a_lid += 1)
  {
    int a_cond = ((a_lid < 2) || (a_gid != 1408));
    if (a_cond)
    {
      int a = ((32 * a_lid) + a_tid);
      int n2_cond = (n2_tid < 7);
      if (n2_cond)
      {
        float LX_T2251 = agg[a_lid];
        int gout_idx = (((a_gid + a) + (10528 * n1_gid)) + (1504 * n2_tid));
        X_T2251[gout_idx] = LX_T2251;
      }
    }
  }
}
