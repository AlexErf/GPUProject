#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T2168[n0, n1, n2, a : _T3122, _T3123, _T3124, _T3125] = =(X_T2167[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T2168[n0, n1, n2, a : _T3122, _T3123, _T3124, _T3125] = =(X_T2167[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1632, 0 <= a < 1664, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 1632 }
// Defracted:
// X_T2168[n0, n1, n2, a : _T3122, _T3123, _T3124, _T3125] = =(X_T2167[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T2168   X_T2167  
//        a      1632         1         1  
//       n1         7     11648     11424  
//       n2         7      1664      1632  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1632, 7, 7 }
// Out stride: { 1, 11648, 1664 }
// Input 1 offset: 0
// Input 1 stride: { 1, 11424, 1632 }
// Tile size: { 1632, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Computed true ops: 159936
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 6528
// Computed out regs: 7168
// Computed mem read: 6528
// Computed mem write: 6528
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_753(__global float* restrict  X_T2168, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1632];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 1632) + (n1_gid * 11424));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 7; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 6) || (a_n2_tid < 96));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)79967)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 7; a_lid += 1)
    {
      int a_cond = ((a_lid < 6) || (a_tid < 96));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 7; a_lid += 1)
  {
    int a_cond = ((a_lid < 6) || (a_tid < 96));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T2168 = agg[a_lid];
      int gout_idx = ((a + (11648 * n1_gid)) + (1664 * n2_gid));
      X_T2168[gout_idx] = LX_T2168;
    }
  }
}
