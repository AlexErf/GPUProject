#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 23 1
// lid: 256 1 1
// Original:
// X_T248[n, x0, x1, g, gco : _T253, _T254, _T255, _T256, _T257] = +(X_T247[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T61[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T248[n, x0, x1, g, gco : _T253, _T254, _T255, _T256, _T257] = +(X_T247[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T61[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 19, 0 <= x1 < 19, 0 <= -1 + k0 + x0 < 19, 0 <= -1 + k1 + x1 < 19, 0 <= g < 728, 0 <= g + gci < 728, 0 <= g < 728, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 19, 0 <= x1 < 19, 0 <= -1 + k0 + x0 < 19, 0 <= -1 + k1 + x1 < 19, 0 <= g < 728, 0 <= g + gci < 728 }
// Defracted:
// X_T248[n, x0, x1, g, gco : _T253, _T254, _T255, _T256, _T257] = +(X_T247[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T61[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T248    X_T247     X_T61  
//        g       728         1         1         1  
//       k0         3         0     13832      2184  
//       k1         3         0       728       728  
//       x0        19     13832     13832         0  
//       x1        19       728       728         0  
//      off                   0    -14560         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 19
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 19
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 728, 3, 3, 19, 19 }
// Out stride: { 1, 0, 0, 13832, 728 }
// Input 1 offset: -14560
// Input 1 stride: { 1, 13832, 728, 13832, 728 }
// Input 2 offset: 0
// Input 2 stride: { 1, 2184, 728, 0, 0 }
// Tile size: { 32, 3, 3, 4, 19 }
// Contraction output var shape: fp32(1, 19, 19, 728, 1):(262808, 13832, 728, 1, 1):1026.59 KiB
// Computed true ops: 4730544
// Computed work groups: 115
// Computed inner loops: 1
// Computed shared mem: 16128
// Computed out regs: 10240
// Computed mem read: 16000
// Computed mem write: 9728
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 23, 1
__kernel void kernel_c28_sdk_76(__global float* restrict  X_T248, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -14560);
  int tid = get_local_id(0);
  float agg[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3744];
  __local float in2_shared[288];
  int g_gid = (get_group_id(1) * 32);
  int x0_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 728)) + (k0_gid * 13832)) + (x0_gid * 13832));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 15; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 14) || (k1_x1_k0_x0_tid < 4));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((117 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (728 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)14560, (int)277367)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 728)) + (k0_gid * 2184));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (728 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)6551)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 19)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 18) <= 19)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x1_lid = 0; x1_lid < 10; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 9) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((117 * g_tid) + k1_lid) + x1) + (19 * k0_lid)) + (19 * x0_tid))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x1_lid = 0; x1_lid < 10; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 9) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((117 * g_tid) + k1_lid) + x1) + (19 * k0_lid)) + (19 * x0_tid))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)(x1_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0_tid))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0_tid)) <= 19)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 19))));
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
  int g_cond = ((g_gid != 704) || (g_tid < 24));
  if (g_cond)
  {
    int x0_cond = ((x0_gid != 16) || (x0_tid < 3));
    if (x0_cond)
    {
      for (int x1_lid = 0; x1_lid < 10; x1_lid += 1)
      {
        int x1_cond = ((x1_lid < 9) || (x1_tid < 1));
        if (x1_cond)
        {
          int x1 = ((2 * x1_lid) + x1_tid);
          float LX_T248 = agg[x1_lid];
          int gout_idx = (((g_gid + g_tid) + (13832 * (x0_gid + x0_tid))) + (728 * x1));
          if (((gout_idx >= 0) && (gout_idx < 262808)))
          {
            X_T248[gout_idx] = LX_T248;
          }
        }
      }
    }
  }
}
