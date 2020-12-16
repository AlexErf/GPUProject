#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Original:
// X_T1925[n0, n1, n2, 1008 + a : _T3039, _T3040, _T3041, _T3042] = =(X_T1924[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1925[n0, n1, n2, 1008 + a : _T3039, _T3040, _T3041, _T3042] = =(X_T1924[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= a < 336, 0 <= 1008 + a < 2016, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 21, 0 <= n2 < 21, 0 <= a < 336 }
// Defracted:
// X_T1925[n0, n1, n2, 1008 + a : _T3039, _T3040, _T3041, _T3042] = =(X_T1924[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1925   X_T1924  
//        a       336         1         1  
//       n1        21     42336      7056  
//       n2        21      2016       336  
//      off                1008         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 336, 21, 21 }
// Out stride: { 1, 42336, 2016 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7056, 336 }
// Tile size: { 32, 2, 8 }
// Contraction output var shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Computed true ops: 296352
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 2120
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_734(__global float* restrict  X_T1925, __global const float* restrict  in1)
{
  X_T1925 = (X_T1925 + 1008);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[530];
  int a_gid = (get_group_id(1) * 32);
  int n2_gid = (get_group_id(0) * 8);
  int n1_gid = (get_group_id(2) * 2);
  {
    {
      int gbase = ((a_gid + (n2_gid * 336)) + (n1_gid * 7056));
      int a_tid = (tid % 32);
      int n2_tid = ((tid / 32) % 8);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int lidx = ((a_tid + (33 * n2_tid)) + (265 * n1_lid));
        int gidx = (((gbase + a_tid) + (336 * n2_tid)) + (7056 * n1_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)148175)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 2; n2_lid += 1)
    {
      int n2 = ((4 * n2_lid) + n2_tid);
      float val1 = in1_shared[((a_tid + (33 * n2)) + (265 * n1_tid))];
      agg[n2_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  int a_cond = ((a_gid != 320) || (a_tid < 16));
  if (a_cond)
  {
    for (int n2_lid = 0; n2_lid < 2; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 1) || ((n2_gid != 16) || (n2_tid < 1)));
      if (n2_cond)
      {
        int n2 = ((4 * n2_lid) + n2_tid);
        int n1_cond = ((n1_gid != 20) || (n1_tid < 1));
        if (n1_cond)
        {
          float LX_T1925 = agg[n2_lid];
          int gout_idx = (((a_gid + a_tid) + (42336 * (int)(n1_gid + n1_tid))) + (2016 * (n2_gid + n2)));
          X_T1925[gout_idx] = LX_T1925;
        }
      }
    }
  }
}
