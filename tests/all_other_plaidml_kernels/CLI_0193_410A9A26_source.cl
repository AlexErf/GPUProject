#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T1284[n, x0, x1, g, gco : _T2012, _T2013, _T2014, _T2015, _T2016] = +(X_T1283[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1267[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T1284[n, x0, x1, g, gco : _T2012, _T2013, _T2014, _T2015, _T2016] = +(X_T1283[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1267[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -2 + k0 + x0 < 14, 0 <= -2 + k1 + x1 < 14, 0 <= g < 88, 0 <= g + gci < 88, 0 <= g < 88, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -2 + k0 + x0 < 14, 0 <= -2 + k1 + x1 < 14, 0 <= g < 88, 0 <= g + gci < 88 }
// Defracted:
// X_T1284[n, x0, x1, g, gco : _T2012, _T2013, _T2014, _T2015, _T2016] = +(X_T1283[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1267[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1284   X_T1283   X_T1267  
//        g        88         1         1         1  
//       k0         5         0      1232       440  
//       k1         5         0        88        88  
//       x0        14      1232      1232         0  
//       x1        14        88        88         0  
//      off                   0     -2640         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 15
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 15
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 88, 5, 5, 14, 14 }
// Out stride: { 1, 0, 0, 1232, 88 }
// Input 1 offset: -2640
// Input 1 stride: { 1, 1232, 88, 1232, 88 }
// Input 2 offset: 0
// Input 2 stride: { 1, 440, 88, 0, 0 }
// Tile size: { 32, 5, 5, 14, 2 }
// Contraction output var shape: fp32(1, 14, 14, 88, 1):(17248, 1232, 88, 1, 1):67.375 KiB
// Computed true ops: 862400
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 17480
// Computed out regs: 4096
// Computed mem read: 17024
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_482(__global float* restrict  X_T1284, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -2640);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3570];
  __local float in2_shared[800];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = (((g_gid + (k1_gid * 88)) + (x1_gid * 88)) + (k0_gid * 1232));
        int g_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 8);
        int k1_x1_cond = (k1_x1_tid < 6);
        if (k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 18; k0_x0_lid += 1)
          {
            int lidx = ((g_tid + (595 * k1_x1_tid)) + (33 * k0_x0_lid));
            int gidx = (((gbase + g_tid) + (88 * k1_x1_tid)) + (1232 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)2640, (int)19887)];
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
      if ((((((-1 * k0_gid) <= -2) && ((((k0_gid + 5) - 1) + 13) <= 15)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -2)) && ((((k1_gid + 5) - 1) + ((x1_gid + 2) - 1)) <= 15)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[((((g_tid + (595 * k1_lid)) + (595 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[((((g_tid + (595 * k1_lid)) + (595 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -2) && (((k0_gid + k0_lid) + x0) <= 15)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -2)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 15))));
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int g_cond = ((g_gid != 64) || (g_tid < 24));
  if (g_cond)
  {
    for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
      if (x0_cond)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T1284 = agg[x0_lid];
        int gout_idx = (((g_gid + g_tid) + (1232 * x0)) + (88 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 17248)))
        {
          X_T1284[gout_idx] = LX_T1284;
        }
      }
    }
  }
}
