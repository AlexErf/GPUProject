#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 2 1
// lid: 256 1 1
// Original:
// X_T2866[n, x0, x1, g, gco : _T4550, _T4551, _T4552, _T4553, _T4554] = +(X_T2865[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2839[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T2866[n, x0, x1, g, gco : _T4550, _T4551, _T4552, _T4553, _T4554] = +(X_T2865[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2839[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -3 + k0 + x0 < 11, 0 <= -3 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672, 0 <= g < 672, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -3 + k0 + x0 < 11, 0 <= -3 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672 }
// Defracted:
// X_T2866[n, x0, x1, g, gco : _T4550, _T4551, _T4552, _T4553, _T4554] = +(X_T2865[n, -3 + k0 + x0, -3 + k1 + x1, g + gci] * X_T2839[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2866   X_T2865   X_T2839  
//        g       672         1         1         1  
//       k0         7         0      7392      4704  
//       k1         7         0       672       672  
//       x0        11      7392      7392         0  
//       x1        11       672       672         0  
//      off                   0    -24192         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -3
// Constraint: (0,1,0,1,0) <= 13
// Constraint: (0,0,-1,0,-1) <= -3
// Constraint: (0,0,1,0,1) <= 13
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 672, 7, 7, 11, 11 }
// Out stride: { 1, 0, 0, 7392, 672 }
// Input 1 offset: -24192
// Input 1 stride: { 1, 7392, 672, 7392, 672 }
// Input 2 offset: 0
// Input 2 stride: { 1, 4704, 672, 0, 0 }
// Tile size: { 32, 7, 7, 11, 8 }
// Contraction output var shape: fp32(1, 11, 11, 672, 1):(81312, 7392, 672, 1, 1):317.625 KiB
// Computed true ops: 7968576
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 30720
// Computed out regs: 12288
// Computed mem read: 30592
// Computed mem write: 11264
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 2, 1
__kernel void kernel_c42_sdk_1107(__global float* restrict  X_T2866, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -24192);
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6112];
  __local float in2_shared[1568];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 7; k0_gid += 7)
  {
    for (int k1_gid = 0; k1_gid < 7; k1_gid += 7)
    {
      {
        int gbase = (((g_gid + (k1_gid * 672)) + (x1_gid * 672)) + (k0_gid * 7392));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 24; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 23) || (k1_x1_k0_x0_tid < 6));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((191 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (672 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)24192, (int)105503)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 672)) + (k0_gid * 4704));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 7; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 6) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((49 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (672 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)32927)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -3) && ((((k0_gid + 7) - 1) + 10) <= 13)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -3)) && ((((k1_gid + 7) - 1) + ((x1_gid + 8) - 1)) <= 13)))
      {
        for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
              {
                int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
                int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
                float val1 = in1_shared[(((((191 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0))];
                float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
              }
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
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
              {
                int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
                int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
                float val1 = in1_shared[(((((191 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0))];
                float val2 = in2_shared[(((49 * g_tid) + k1_lid) + (7 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -3) && (((k0_gid + k0_lid) + x0) <= 13)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -3)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 13))));
              }
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
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 8) || (x1_tid < 3))) && ((x1_lid < 1) || (x1_gid != 8)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T2866 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = (((g_gid + g_tid) + (7392 * x0)) + (672 * (x1_gid + x1)));
          if (((gout_idx >= 0) && (gout_idx < 81312)))
          {
            X_T2866[gout_idx] = LX_T2866;
          }
        }
      }
    }
  }
}
