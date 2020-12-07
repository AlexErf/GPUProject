#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4608 2 1
// lid: 256 1 1
// Original:
// X_T493[n, x0, x1, g, gco : _T574, _T575, _T576, _T577, _T578] = +(X_T492[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T479[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T493[n, x0, x1, g, gco : _T574, _T575, _T576, _T577, _T578] = +(X_T492[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T479[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= g < 576, 0 <= g + gci < 576, 0 <= g < 576, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= g < 576, 0 <= g + gci < 576 }
// Defracted:
// X_T493[n, x0, x1, g, gco : _T574, _T575, _T576, _T577, _T578] = +(X_T492[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T479[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T493    X_T492    X_T479  
//        g       576         1         1         1  
//       k0         3         0      8064      1728  
//       k1         3         0       576       576  
//       x0        14      8064      8064         0  
//       x1        14       576       576         0  
//      off                   0     -8640         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 14
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 14
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 576, 3, 3, 14, 14 }
// Out stride: { 1, 0, 0, 8064, 576 }
// Input 1 offset: -8640
// Input 1 stride: { 1, 8064, 576, 8064, 576 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1728, 576, 0, 0 }
// Tile size: { 32, 3, 3, 8, 14 }
// Contraction output var shape: fp32(1, 14, 14, 576, 1):(112896, 8064, 576, 1, 1):441 KiB
// Computed true ops: 2032128
// Computed work groups: 36
// Computed inner loops: 1
// Computed shared mem: 19456
// Computed out regs: 14336
// Computed mem read: 19328
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4608, 2, 1
__kernel void kernel_c43_sdk_131(__global float* restrict  X_T493, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -8640);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4576];
  __local float in2_shared[288];
  int g_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 576)) + (k0_gid * 8064)) + (x0_gid * 8064));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 18; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 17) || (k1_x1_k0_x0_tid < 6));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((143 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (576 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)8640, (int)121535)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 576)) + (k0_gid * 1728));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (576 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)5183)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 14)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 13) <= 14)))
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
              for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
              {
                int x1 = ((2 * x1_lid) + x1_tid);
                float val1 = in1_shared[(((((143 * g_tid) + k1_lid) + x1) + (14 * k0_lid)) + (14 * x0))];
                float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 7));
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
              for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
              {
                int x1 = ((2 * x1_lid) + x1_tid);
                float val1 = in1_shared[(((((143 * g_tid) + k1_lid) + x1) + (14 * k0_lid)) + (14 * x0))];
                float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 7));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 14)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 14)));
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
      for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
      {
        int x1 = ((2 * x1_lid) + x1_tid);
        float LX_T493 = agg[(x1_lid + (x0_lid * 7))];
        int gout_idx = (((g_gid + g_tid) + (8064 * (x0_gid + x0))) + (576 * x1));
        if (((gout_idx >= 0) && (gout_idx < 112896)))
        {
          X_T493[gout_idx] = LX_T493;
        }
      }
    }
  }
}
