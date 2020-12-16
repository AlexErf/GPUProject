#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 6 11
// lid: 256 1 1
// Original:
// X_T523[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T522[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T511[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T523[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T522[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T511[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -2 + k0 + x0 < 42, 0 <= -2 + k1 + x1 < 42, 0 <= g < 168, 0 <= g + gci < 168, 0 <= g < 168, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -2 + k0 + x0 < 42, 0 <= -2 + k1 + x1 < 42, 0 <= g < 168, 0 <= g + gci < 168 }
// Defracted:
// X_T523[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T522[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T511[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T523    X_T522    X_T511  
//        g       168         1         1         1  
//       k0         5         0      7056       840  
//       k1         5         0       168       168  
//       x0        42      7056      7056         0  
//       x1        42       168       168         0  
//      off                   0    -14448         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 43
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 43
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 168, 5, 5, 42, 42 }
// Out stride: { 1, 0, 0, 7056, 168 }
// Input 1 offset: -14448
// Input 1 stride: { 1, 7056, 168, 7056, 168 }
// Input 2 offset: 0
// Input 2 stride: { 1, 840, 168, 0, 0 }
// Tile size: { 32, 5, 5, 16, 4 }
// Contraction output var shape: fp32(1, 42, 42, 168, 1):(296352, 7056, 168, 1, 1):1157.62 KiB
// Computed true ops: 14817600
// Computed work groups: 198
// Computed inner loops: 1
// Computed shared mem: 24352
// Computed out regs: 8192
// Computed mem read: 23680
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 6, 11
__kernel void kernel_c42_sdk_187(__global float* restrict  X_T523, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -14448);
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[5288];
  __local float in2_shared[800];
  int g_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(2) * 4);
  int x0_gid = (get_group_id(0) * 16);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = ((((g_gid + (k1_gid * 168)) + (x1_gid * 168)) + (k0_gid * 7056)) + (x0_gid * 7056));
        int g_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 20; k0_x0_lid += 1)
        {
          int lidx = ((g_tid + (661 * k1_x1_tid)) + (33 * k0_x0_lid));
          int gidx = (((gbase + g_tid) + (168 * k1_x1_tid)) + (7056 * k0_x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)14448, (int)310799)];
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 168)) + (k0_gid * 840));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 4; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 3) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((25 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (168 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)4199)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -2) && ((((k0_gid + 5) - 1) + ((x0_gid + 16) - 1)) <= 43)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -2)) && ((((k1_gid + 5) - 1) + ((x1_gid + 4) - 1)) <= 43)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((g_tid + (661 * k1_lid)) + (661 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = agg_rhs;
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
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((g_tid + (661 * k1_lid)) + (661 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -2) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 43)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -2)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 43)));
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
  int g_cond = ((g_gid != 160) || (g_tid < 8));
  if (g_cond)
  {
    int x1_cond = ((x1_gid != 40) || (x1_tid < 2));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 5) || (x0_gid != 32));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T523 = agg[x0_lid];
          int gout_idx = (((g_gid + g_tid) + (7056 * (x0_gid + x0))) + (168 * (x1_gid + x1_tid)));
          if (((gout_idx >= 0) && (gout_idx < 296352)))
          {
            X_T523[gout_idx] = LX_T523;
          }
        }
      }
    }
  }
}
