#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 4 1
// lid: 256 1 1
// Original:
// X_T1246[n, x0, x1, g, gco : _T1946, _T1947, _T1948, _T1949, _T1950] = +(X_T1245[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T1218[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T1246[n, x0, x1, g, gco : _T1946, _T1947, _T1948, _T1949, _T1950] = +(X_T1245[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T1218[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -3 + k0 + x0 < 14, 0 <= -3 + k1 + x1 < 14, 0 <= g < 88, 0 <= g + gci < 88, 0 <= g < 88, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -3 + k0 + x0 < 14, 0 <= -3 + k1 + x1 < 14, 0 <= g < 88, 0 <= g + gci < 88 }
// Defracted:
// X_T1246[n, x0, x1, g, gco : _T1946, _T1947, _T1948, _T1949, _T1950] = +(X_T1245[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T1218[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1246   X_T1245   X_T1218  
//        g        88         1         1         1  
//       k0         7         0      1232       616  
//       k1         7         0        88        88  
//       x0        14      1232      1232         0  
//       x1        14        88        88         0  
//      off                   0     -3960         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -3
// Constraint: (0,1,0,1,0) <= 16
// Constraint: (0,0,-1,0,-1) <= -3
// Constraint: (0,0,1,0,1) <= 16
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 88, 7, 7, 14, 14 }
// Out stride: { 1, 0, 0, 1232, 88 }
// Input 1 offset: -3960
// Input 1 stride: { 1, 1232, 88, 1232, 88 }
// Input 2 offset: 0
// Input 2 stride: { 1, 616, 88, 0, 0 }
// Tile size: { 32, 7, 7, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 88, 1):(17248, 1232, 88, 1, 1):67.375 KiB
// Computed true ops: 1690304
// Computed work groups: 12
// Computed inner loops: 1
// Computed shared mem: 32712
// Computed out regs: 7168
// Computed mem read: 31872
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 4, 1
__kernel void kernel_c42_sdk_467(__global float* restrict  X_T1246, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -3960);
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6610];
  __local float in2_shared[1568];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 7; k0_gid += 7)
  {
    for (int k1_gid = 0; k1_gid < 7; k1_gid += 7)
    {
      {
        int gbase = (((g_gid + (k1_gid * 88)) + (x1_gid * 88)) + (k0_gid * 1232));
        int g_tid = (tid % 32);
        int k0_x0_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 3; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 2) || (k0_x0_tid < 4));
          if (k0_x0_cond)
          {
            int k0_x0 = ((8 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 10; k1_x1_lid += 1)
            {
              int lidx = ((g_tid + (33 * k0_x0)) + (661 * k1_x1_lid));
              int gidx = (((gbase + g_tid) + (1232 * k0_x0)) + (88 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)3960, (int)21207)];
            }
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 88)) + (k0_gid * 616));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 7; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 6) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((49 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (88 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)4311)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -3) && ((((k0_gid + 7) - 1) + 13) <= 16)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -3)) && ((((k1_gid + 7) - 1) + ((x1_gid + 4) - 1)) <= 16)))
      {
        for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((g_tid + (661 * k1_lid)) + (661 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = agg_rhs;
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((g_tid + (661 * k1_lid)) + (661 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -3) && (((k0_gid + k0_lid) + x0) <= 16)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -3)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 16)));
            }
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
    int x1_cond = ((x1_gid != 12) || (x1_tid < 2));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T1246 = agg[x0_lid];
        int gout_idx = (((g_gid + g_tid) + (1232 * x0)) + (88 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 17248)))
        {
          X_T1246[gout_idx] = LX_T1246;
        }
      }
    }
  }
}
