#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T127[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T126[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T110[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T127[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T126[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T110[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 42, 0 <= g + gci < 42, 0 <= g < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= -2 + k0 + x0 < 83, 0 <= -2 + k1 + x1 < 83, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 42, 0 <= g + gci < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= -2 + k0 + x0 < 83, 0 <= -2 + k1 + x1 < 83 }
// Defracted:
// X_T127[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T126[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T110[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T127    X_T126    X_T110  
//        g        42         1         1         1  
//       k0         5         0      3486       210  
//       k1         5         0        42        42  
//       x0        83      3486      3486         0  
//       x1        83        42        42         0  
//      off                   0     -7056         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 84
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 84
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 42, 5, 5, 83, 83 }
// Out stride: { 1, 0, 0, 3486, 42 }
// Input 1 offset: -7056
// Input 1 stride: { 1, 3486, 42, 3486, 42 }
// Input 2 offset: 0
// Input 2 stride: { 1, 210, 42, 0, 0 }
// Tile size: { 42, 5, 5, 8, 8 }
// Contraction output var shape: fp32(1, 83, 83, 42, 1):(289338, 3486, 42, 1, 1):1130.23 KiB
// Computed true ops: 14466900
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 28440
// Computed out regs: 16384
// Computed mem read: 28288
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_30(__global float* restrict  X_T127, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -7056);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6060];
  __local float in2_shared[1050];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = ((((k1_gid * 42) + (x1_gid * 42)) + (k0_gid * 3486)) + (x0_gid * 3486));
        int g_k1_x1_tid = (tid % 256);
        for (int g_k1_x1_lid = 0; g_k1_x1_lid < 2; g_k1_x1_lid += 1)
        {
          int g_k1_x1_cond = ((g_k1_x1_lid < 1) || (g_k1_x1_tid < 248));
          if (g_k1_x1_cond)
          {
            int g_k1_x1 = ((256 * g_k1_x1_lid) + g_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 12; k0_x0_lid += 1)
            {
              int lidx = (g_k1_x1 + (505 * k0_x0_lid));
              int gidx = ((gbase + g_k1_x1) + (3486 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)7056, (int)296393)];
            }
          }
        }
      }
      {
        int gbase = ((k1_gid * 42) + (k0_gid * 210));
        int g_k1_k0_tid = (tid % 256);
        for (int g_k1_k0_lid = 0; g_k1_k0_lid < 5; g_k1_k0_lid += 1)
        {
          int g_k1_k0_cond = ((g_k1_k0_lid < 4) || (g_k1_k0_tid < 26));
          if (g_k1_k0_cond)
          {
            int g_k1_k0 = ((256 * g_k1_k0_lid) + g_k1_k0_tid);
            int gidx = (gbase + g_k1_k0);
            in2_shared[g_k1_k0] = in2[clamp((int)gidx, (int)0, (int)1049)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -2) && ((((k0_gid + 5) - 1) + ((x0_gid + 8) - 1)) <= 84)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -2)) && ((((k1_gid + 5) - 1) + ((x1_gid + 8) - 1)) <= 84)))
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
                  int g_cond = ((g_lid < 1) || (g_tid < 10));
                  int g = select((int)0, (int)((32 * g_lid) + g_tid), (int)g_cond);
                  float val1 = in1_shared[((((g + (42 * k1_lid)) + (42 * x1)) + (505 * k0_lid)) + (505 * x0))];
                  float val2 = in2_shared[((g + (42 * k1_lid)) + (210 * k0_lid))];
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
                  int g_cond = ((g_lid < 1) || (g_tid < 10));
                  int g = select((int)0, (int)((32 * g_lid) + g_tid), (int)g_cond);
                  float val1 = in1_shared[((((g + (42 * k1_lid)) + (42 * x1)) + (505 * k0_lid)) + (505 * x0))];
                  float val2 = in2_shared[((g + (42 * k1_lid)) + (210 * k0_lid))];
                  int agg_idx = ((g_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(g_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -2) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 84)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -2)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 84))));
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
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 80) || (x1_tid < 3))) && ((x1_lid < 1) || (x1_gid != 80)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 1) || ((x0_gid != 80) || (x0_tid < 1))) && ((x0_lid < 2) || (x0_gid != 80)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int g_lid = 0; g_lid < 2; g_lid += 1)
          {
            int g_cond = ((g_lid < 1) || (g_tid < 10));
            if (g_cond)
            {
              int g = ((32 * g_lid) + g_tid);
              float LX_T127 = agg[((g_lid + (x1_lid * 2)) + (x0_lid * 4))];
              int gout_idx = ((g + (3486 * (x0_gid + x0))) + (42 * (x1_gid + x1)));
              if (((gout_idx >= 0) && (gout_idx < 289338)))
              {
                X_T127[gout_idx] = LX_T127;
              }
            }
          }
        }
      }
    }
  }
}
