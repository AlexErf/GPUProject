#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T1271[n, x0, x1, g, gco : _T1992, _T1993, _T1994, _T1995, _T1996] = +(X_T1270[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1268[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T1271[n, x0, x1, g, gco : _T1992, _T1993, _T1994, _T1995, _T1996] = +(X_T1270[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1268[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 31, 0 <= k1 + 2*x1 < 31, 0 <= g < 88, 0 <= g + gci < 88, 0 <= g < 88, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 31, 0 <= k1 + 2*x1 < 31, 0 <= g < 88, 0 <= g + gci < 88 }
// Defracted:
// X_T1271[n, x0, x1, g, gco : _T1992, _T1993, _T1994, _T1995, _T1996] = +(X_T1270[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1268[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1271   X_T1270   X_T1268  
//        g        88         1         1         1  
//       k0         5         0      2728       440  
//       k1         5         0        88        88  
//       x0        14      1232      5456         0  
//       x1        14        88       176         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 88, 5, 5, 14, 14 }
// Out stride: { 1, 0, 0, 1232, 88 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2728, 88, 5456, 176 }
// Input 2 offset: 0
// Input 2 stride: { 1, 440, 88, 0, 0 }
// Tile size: { 32, 5, 5, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 88, 1):(17248, 1232, 88, 1, 1):67.375 KiB
// Computed true ops: 862400
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 30976
// Computed out regs: 4096
// Computed mem read: 30976
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_478(__global float* restrict  X_T1271, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[6944];
  __local float in2_shared[800];
  int g_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = (((g_gid + (k1_gid * 88)) + (k0_gid * 2728)) + (x0_gid * 5456));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 28; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 27) || (k1_x1_k0_x0_tid < 1));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((217 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (88 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)84567)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 88)) + (k0_gid * 440));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 4; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 3) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((25 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (88 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)2199)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
        {
          int g_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
          {
            int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
            int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
            float val1 = in1_shared[(((((217 * g_tid) + k1_lid) + (2 * x1)) + (31 * k0_lid)) + (62 * x0_tid))];
            float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
            float agg_rhs = mad(val2, val1, agg[x1_lid]);
            agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int g_cond = ((g_gid != 64) || (g_tid < 24));
  if (g_cond)
  {
    for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
      if (x1_cond)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        float LX_T1271 = agg[x1_lid];
        int gout_idx = (((g_gid + g_tid) + (1232 * (x0_gid + x0_tid))) + (88 * x1));
        X_T1271[gout_idx] = LX_T1271;
      }
    }
  }
}
