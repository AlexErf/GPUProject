#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 1 1
// lid: 256 1 1
// Original:
// X_T124[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T123[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T107[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T124[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T123[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T107[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 11, 0 <= g + gci < 11, 0 <= g < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -2 + k0 + x0 < 56, 0 <= -2 + k1 + x1 < 56, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 11, 0 <= g + gci < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= -2 + k0 + x0 < 56, 0 <= -2 + k1 + x1 < 56 }
// Defracted:
// X_T124[n, x0, x1, g, gco : _T141, _T142, _T143, _T144, _T145] = +(X_T123[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T107[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T124    X_T123    X_T107  
//        g        11         1         1         1  
//       k0         5         0       616        55  
//       k1         5         0        11        11  
//       x0        56       616       616         0  
//       x1        56        11        11         0  
//      off                   0     -1254         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 57
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 57
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 11, 5, 5, 56, 56 }
// Out stride: { 1, 0, 0, 616, 11 }
// Input 1 offset: -1254
// Input 1 stride: { 1, 616, 11, 616, 11 }
// Input 2 offset: 0
// Input 2 stride: { 1, 55, 11, 0, 0 }
// Tile size: { 11, 5, 5, 4, 56 }
// Contraction output var shape: fp32(1, 56, 56, 11, 1):(34496, 616, 11, 1, 1):134.75 KiB
// Computed true ops: 1724800
// Computed work groups: 14
// Computed inner loops: 1
// Computed shared mem: 20988
// Computed out regs: 14336
// Computed mem read: 20864
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 1, 1
__kernel void kernel_c42_sdk_30(__global float* restrict  X_T124, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -1254);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4972];
  __local float in2_shared[275];
  int x0_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = (((k1_gid * 11) + (k0_gid * 616)) + (x0_gid * 616));
        int g_k1_x1_k0_x0_tid = (tid % 256);
        for (int g_k1_x1_k0_x0_lid = 0; g_k1_x1_k0_x0_lid < 20; g_k1_x1_k0_x0_lid += 1)
        {
          int g_k1_x1_k0_x0_cond = ((g_k1_x1_k0_x0_lid < 19) || (g_k1_x1_k0_x0_tid < 108));
          if (g_k1_x1_k0_x0_cond)
          {
            int g_k1_x1_k0_x0 = ((256 * g_k1_x1_k0_x0_lid) + g_k1_x1_k0_x0_tid);
            int gidx = (gbase + g_k1_x1_k0_x0);
            in1_shared[g_k1_x1_k0_x0] = in1[clamp((int)gidx, (int)1254, (int)35749)];
          }
        }
      }
      {
        int gbase = ((k1_gid * 11) + (k0_gid * 55));
        int g_k1_k0_tid = (tid % 256);
        for (int g_k1_k0_lid = 0; g_k1_k0_lid < 2; g_k1_k0_lid += 1)
        {
          int g_k1_k0_cond = ((g_k1_k0_lid < 1) || (g_k1_k0_tid < 19));
          if (g_k1_k0_cond)
          {
            int g_k1_k0 = ((256 * g_k1_k0_lid) + g_k1_k0_tid);
            int gidx = (gbase + g_k1_k0);
            in2_shared[g_k1_k0] = in2[clamp((int)gidx, (int)0, (int)274)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -2) && ((((k0_gid + 5) - 1) + ((x0_gid + 4) - 1)) <= 57)) && ((-1 * k1_gid) <= -2)) && ((((k1_gid + 5) - 1) + 55) <= 57)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int g_cond = (g_tid < 11);
            int g = select((int)0, (int)g_tid, (int)g_cond);
            for (int x1_lid = 0; x1_lid < 14; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              float val1 = in1_shared[((((g + (11 * k1_lid)) + (11 * x1)) + (616 * k0_lid)) + (616 * x0_tid))];
              float val2 = in2_shared[((g + (11 * k1_lid)) + (55 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)g_cond);
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
            int g_tid = (tid % 16);
            int x1_tid = ((tid / 16) % 4);
            int x0_tid = ((tid / 64) % 4);
            int g_cond = (g_tid < 11);
            int g = select((int)0, (int)g_tid, (int)g_cond);
            for (int x1_lid = 0; x1_lid < 14; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              float val1 = in1_shared[((((g + (11 * k1_lid)) + (11 * x1)) + (616 * k0_lid)) + (616 * x0_tid))];
              float val2 = in2_shared[((g + (11 * k1_lid)) + (55 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)(g_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0_tid))) <= -2) && (((k0_gid + k0_lid) + (x0_gid + x0_tid)) <= 57)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -2)) && (((k1_gid + k1_lid) + x1) <= 57))));
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 4);
  int x0_tid = ((tid / 64) % 4);
  int g_cond = (g_tid < 11);
  if (g_cond)
  {
    for (int x1_lid = 0; x1_lid < 14; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      float LX_T124 = agg[x1_lid];
      int gout_idx = ((g_tid + (616 * (x0_gid + x0_tid))) + (11 * x1));
      if (((gout_idx >= 0) && (gout_idx < 34496)))
      {
        X_T124[gout_idx] = LX_T124;
      }
    }
  }
}
