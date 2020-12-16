#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 2
// lid: 256 1 1
// Original:
// X_T343[n, x0, x1, g, gco : _T384, _T385, _T386, _T387, _T388] = +(X_T342[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T329[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T343[n, x0, x1, g, gco : _T384, _T385, _T386, _T387, _T388] = +(X_T342[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T329[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= g < 384, 0 <= g + gci < 384, 0 <= g < 384, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= g < 384, 0 <= g + gci < 384 }
// Defracted:
// X_T343[n, x0, x1, g, gco : _T384, _T385, _T386, _T387, _T388] = +(X_T342[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T329[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T343    X_T342    X_T329  
//        g       384         1         1         1  
//       k0         3         0      5376      1152  
//       k1         3         0       384       384  
//       x0        14      5376      5376         0  
//       x1        14       384       384         0  
//      off                   0     -5760         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 14
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 14
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 384, 3, 3, 14, 14 }
// Out stride: { 1, 0, 0, 5376, 384 }
// Input 1 offset: -5760
// Input 1 stride: { 1, 5376, 384, 5376, 384 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1152, 384, 0, 0 }
// Tile size: { 128, 3, 3, 8, 2 }
// Contraction output var shape: fp32(1, 14, 14, 384, 1):(75264, 5376, 384, 1, 1):294 KiB
// Computed true ops: 1354752
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 25264
// Computed out regs: 8192
// Computed mem read: 25088
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 2
__kernel void kernel_c43_sdk_88(__global float* restrict  X_T343, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -5760);
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[5164];
  __local float in2_shared[1152];
  int g_gid = (get_group_id(0) * 128);
  int x1_gid = (get_group_id(1) * 2);
  int x0_gid = (get_group_id(2) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((g_gid + (k1_gid * 384)) + (x1_gid * 384)) + (k0_gid * 5376)) + (x0_gid * 5376));
        int g_tid = (tid % 128);
        int k1_x1_tid = ((tid / 128) % 2);
        for (int k1_x1_lid = 0; k1_x1_lid < 2; k1_x1_lid += 1)
        {
          int k1_x1 = ((2 * k1_x1_lid) + k1_x1_tid);
          for (int k0_x0_lid = 0; k0_x0_lid < 10; k0_x0_lid += 1)
          {
            int lidx = ((g_tid + (1291 * k1_x1)) + (129 * k0_x0_lid));
            int gidx = (((gbase + g_tid) + (384 * k1_x1)) + (5376 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)5760, (int)81023)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 384)) + (k0_gid * 1152));
        int g_tid = (tid % 128);
        int k1_k0_tid = ((tid / 128) % 2);
        for (int k1_k0_lid = 0; k1_k0_lid < 5; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 4) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((2 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (384 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)3455)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 14)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 2) - 1)) <= 14)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              for (int g_lid = 0; g_lid < 4; g_lid += 1)
              {
                int g = ((32 * g_lid) + g_tid);
                float val1 = in1_shared[((((g + (1291 * k1_lid)) + (1291 * x1_tid)) + (129 * k0_lid)) + (129 * x0))];
                float val2 = in2_shared[(((9 * g) + k1_lid) + (3 * k0_lid))];
                int agg_idx = (g_lid + (x0_lid * 4));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = agg_rhs;
              }
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
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              for (int g_lid = 0; g_lid < 4; g_lid += 1)
              {
                int g = ((32 * g_lid) + g_tid);
                float val1 = in1_shared[((((g + (1291 * k1_lid)) + (1291 * x1_tid)) + (129 * k0_lid)) + (129 * x0))];
                float val2 = in2_shared[(((9 * g) + k1_lid) + (3 * k0_lid))];
                int agg_idx = (g_lid + (x0_lid * 4));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 14)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 14)));
              }
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
  for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 1) || ((x0_gid != 8) || (x0_tid < 2)));
    if (x0_cond)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      for (int g_lid = 0; g_lid < 4; g_lid += 1)
      {
        int g = ((32 * g_lid) + g_tid);
        float LX_T343 = agg[(g_lid + (x0_lid * 4))];
        int gout_idx = (((g_gid + g) + (5376 * (x0_gid + x0))) + (384 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 75264)))
        {
          X_T343[gout_idx] = LX_T343;
        }
      }
    }
  }
}
