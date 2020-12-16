#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 4 1
// lid: 256 1 1
// Original:
// X_T524[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T523[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T512[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T524[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T523[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T512[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -2 + k0 + x0 < 28, 0 <= -2 + k1 + x1 < 28, 0 <= g < 44, 0 <= g + gci < 44, 0 <= g < 44, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -2 + k0 + x0 < 28, 0 <= -2 + k1 + x1 < 28, 0 <= g < 44, 0 <= g + gci < 44 }
// Defracted:
// X_T524[n, x0, x1, g, gco : _T801, _T802, _T803, _T804, _T805] = +(X_T523[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T512[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T524    X_T523    X_T512  
//        g        44         1         1         1  
//       k0         5         0      1232       220  
//       k1         5         0        44        44  
//       x0        28      1232      1232         0  
//       x1        28        44        44         0  
//      off                   0     -2552         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 29
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 29
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 44, 5, 5, 28, 28 }
// Out stride: { 1, 0, 0, 1232, 44 }
// Input 1 offset: -2552
// Input 1 stride: { 1, 1232, 44, 1232, 44 }
// Input 2 offset: 0
// Input 2 stride: { 1, 220, 44, 0, 0 }
// Tile size: { 44, 5, 5, 8, 8 }
// Contraction output var shape: fp32(1, 28, 28, 44, 1):(34496, 1232, 44, 1, 1):134.75 KiB
// Computed true ops: 1724800
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 29792
// Computed out regs: 16384
// Computed mem read: 29696
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 4, 1
__kernel void kernel_c42_sdk_187(__global float* restrict  X_T524, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -2552);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6348];
  __local float in2_shared[1100];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = ((((k1_gid * 44) + (x1_gid * 44)) + (k0_gid * 1232)) + (x0_gid * 1232));
        int g_k1_x1_tid = (tid % 256);
        for (int g_k1_x1_lid = 0; g_k1_x1_lid < 3; g_k1_x1_lid += 1)
        {
          int g_k1_x1_cond = ((g_k1_x1_lid < 2) || (g_k1_x1_tid < 16));
          if (g_k1_x1_cond)
          {
            int g_k1_x1 = ((256 * g_k1_x1_lid) + g_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 12; k0_x0_lid += 1)
            {
              int lidx = (g_k1_x1 + (529 * k0_x0_lid));
              int gidx = ((gbase + g_k1_x1) + (1232 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)2552, (int)37047)];
            }
          }
        }
      }
      {
        int gbase = ((k1_gid * 44) + (k0_gid * 220));
        int g_k1_k0_tid = (tid % 256);
        for (int g_k1_k0_lid = 0; g_k1_k0_lid < 5; g_k1_k0_lid += 1)
        {
          int g_k1_k0_cond = ((g_k1_k0_lid < 4) || (g_k1_k0_tid < 76));
          if (g_k1_k0_cond)
          {
            int g_k1_k0 = ((256 * g_k1_k0_lid) + g_k1_k0_tid);
            int gidx = (gbase + g_k1_k0);
            in2_shared[g_k1_k0] = in2[clamp((int)gidx, (int)0, (int)1099)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -2) && ((((k0_gid + 5) - 1) + ((x0_gid + 8) - 1)) <= 29)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -2)) && ((((k1_gid + 5) - 1) + ((x1_gid + 8) - 1)) <= 29)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int g_lid = 0; g_lid < 2; g_lid += 1)
                {
                  int g_cond = ((g_lid < 1) || (g_tid < 12));
                  int g = select((int)0, (int)((32 * g_lid) + g_tid), (int)g_cond);
                  float val1 = in1_shared[((((g + (44 * k1_lid)) + (44 * x1)) + (529 * k0_lid)) + (529 * x0))];
                  float val2 = in2_shared[((g + (44 * k1_lid)) + (220 * k0_lid))];
                  int agg_idx = ((g_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)g_cond);
                }
              }
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
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int g_lid = 0; g_lid < 2; g_lid += 1)
                {
                  int g_cond = ((g_lid < 1) || (g_tid < 12));
                  int g = select((int)0, (int)((32 * g_lid) + g_tid), (int)g_cond);
                  float val1 = in1_shared[((((g + (44 * k1_lid)) + (44 * x1)) + (529 * k0_lid)) + (529 * x0))];
                  float val2 = in2_shared[((g + (44 * k1_lid)) + (220 * k0_lid))];
                  int agg_idx = ((g_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(g_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -2) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 29)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -2)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 29))));
                }
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
    int x1_cond = ((x1_lid < 1) || (x1_gid != 24));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 2) || (x0_gid != 24));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int g_lid = 0; g_lid < 2; g_lid += 1)
          {
            int g_cond = ((g_lid < 1) || (g_tid < 12));
            if (g_cond)
            {
              int g = ((32 * g_lid) + g_tid);
              float LX_T524 = agg[((g_lid + (x1_lid * 2)) + (x0_lid * 4))];
              int gout_idx = ((g + (1232 * (x0_gid + x0))) + (44 * (x1_gid + x1)));
              if (((gout_idx >= 0) && (gout_idx < 34496)))
              {
                X_T524[gout_idx] = LX_T524;
              }
            }
          }
        }
      }
    }
  }
}
