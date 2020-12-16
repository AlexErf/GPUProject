#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 1
// lid: 256 1 1
// Original:
// X_T2903[n, x0, x1, g, gco : _T4616, _T4617, _T4618, _T4619, _T4620] = +(X_T2902[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T2887[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T2903[n, x0, x1, g, gco : _T4616, _T4617, _T4618, _T4619, _T4620] = +(X_T2902[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T2887[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -2 + k0 + x0 < 11, 0 <= -2 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672, 0 <= g < 672, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -2 + k0 + x0 < 11, 0 <= -2 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672 }
// Defracted:
// X_T2903[n, x0, x1, g, gco : _T4616, _T4617, _T4618, _T4619, _T4620] = +(X_T2902[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T2887[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2903   X_T2902   X_T2887  
//        g       672         1         1         1  
//       k0         5         0      7392      3360  
//       k1         5         0       672       672  
//       x0        11      7392      7392         0  
//       x1        11       672       672         0  
//      off                   0    -16128         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 12
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 12
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 672, 5, 5, 11, 11 }
// Out stride: { 1, 0, 0, 7392, 672 }
// Input 1 offset: -16128
// Input 1 stride: { 1, 7392, 672, 7392, 672 }
// Input 2 offset: 0
// Input 2 stride: { 1, 3360, 672, 0, 0 }
// Tile size: { 32, 5, 5, 4, 11 }
// Contraction output var shape: fp32(1, 11, 11, 672, 1):(81312, 7392, 672, 1, 1):317.625 KiB
// Computed true ops: 4065600
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 15104
// Computed out regs: 6144
// Computed mem read: 14976
// Computed mem write: 5632
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 1
__kernel void kernel_c42_sdk_1122(__global float* restrict  X_T2903, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -16128);
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2976];
  __local float in2_shared[800];
  int g_gid = (get_group_id(1) * 32);
  int x0_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = (((g_gid + (k1_gid * 672)) + (k0_gid * 7392)) + (x0_gid * 7392));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 12; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 11) || (k1_x1_k0_x0_tid < 4));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((93 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (672 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)16128, (int)97439)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 672)) + (k0_gid * 3360));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 4; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 3) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((25 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (672 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)16799)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -2) && ((((k0_gid + 5) - 1) + ((x0_gid + 4) - 1)) <= 12)) && ((-1 * k1_gid) <= -2)) && ((((k1_gid + 5) - 1) + 10) <= 12)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((93 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0_tid))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
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
            for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((93 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0_tid))];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)(x1_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0_tid))) <= -2) && (((k0_gid + k0_lid) + (x0_gid + x0_tid)) <= 12)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -2)) && (((k1_gid + k1_lid) + x1) <= 12))));
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
  int x0_cond = ((x0_gid != 8) || (x0_tid < 3));
  if (x0_cond)
  {
    for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
      if (x1_cond)
      {
        int x1 = ((2 * x1_lid) + x1_tid);
        float LX_T2903 = agg[x1_lid];
        int gout_idx = (((g_gid + g_tid) + (7392 * (x0_gid + x0_tid))) + (672 * x1));
        if (((gout_idx >= 0) && (gout_idx < 81312)))
        {
          X_T2903[gout_idx] = LX_T2903;
        }
      }
    }
  }
}
