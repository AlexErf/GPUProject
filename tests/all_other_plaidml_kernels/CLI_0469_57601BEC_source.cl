#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 14 1
// lid: 256 1 1
// Original:
// X_T1281[n0, n1, n2, a : _T1796, _T1797, _T1798, _T1799] = =(X_T1280[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1281[n0, n1, n2, a : _T1796, _T1797, _T1798, _T1799] = =(X_T1280[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1152, 0 <= a < 1184, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1152 }
// Defracted:
// X_T1281[n0, n1, n2, a : _T1796, _T1797, _T1798, _T1799] = =(X_T1280[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1281   X_T1280  
//        a      1152         1         1  
//       n1        14     16576     16128  
//       n2        14      1184      1152  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1152, 14, 14 }
// Out stride: { 1, 16576, 1184 }
// Input 1 offset: 0
// Input 1 stride: { 1, 16128, 1152 }
// Tile size: { 128, 1, 14 }
// Contraction output var shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Computed true ops: 451584
// Computed work groups: 126
// Computed inner loops: 1
// Computed shared mem: 7224
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 14, 1
__kernel void kernel_c124_sdk_432(__global float* restrict  X_T1281, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1806];
  int a_gid = (get_group_id(0) * 128);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 16128));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 7; n2_n1_lid += 1)
      {
        int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
        int lidx = (a_tid + (129 * n2_n1));
        int gidx = ((gbase + a_tid) + (1152 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)225791)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int n2_lid = 0; n2_lid < 2; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 1) || (n2_tid < 6));
      int n2 = select((int)0, (int)((8 * n2_lid) + n2_tid), (int)n2_cond);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (129 * n2))];
        int agg_idx = (a_lid + (n2_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n2_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int n2_lid = 0; n2_lid < 2; n2_lid += 1)
  {
    int n2_cond = ((n2_lid < 1) || (n2_tid < 6));
    if (n2_cond)
    {
      int n2 = ((8 * n2_lid) + n2_tid);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T1281 = agg[(a_lid + (n2_lid * 4))];
        int gout_idx = (((a_gid + a) + (16576 * n1_gid)) + (1184 * n2));
        X_T1281[gout_idx] = LX_T1281;
      }
    }
  }
}
