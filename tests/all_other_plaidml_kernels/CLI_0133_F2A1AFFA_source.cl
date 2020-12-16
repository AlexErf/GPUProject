#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4608 1 1
// lid: 256 1 1
// Original:
// X_T570[n, x0, x1, g, gco : _T673, _T674, _T675, _T676, _T677] = +(X_T569[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T23[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T570[n, x0, x1, g, gco : _T673, _T674, _T675, _T676, _T677] = +(X_T569[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T23[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 15, 0 <= k1 + 2*x1 < 15, 0 <= g < 576, 0 <= g + gci < 576, 0 <= g < 576, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 15, 0 <= k1 + 2*x1 < 15, 0 <= g < 576, 0 <= g + gci < 576 }
// Defracted:
// X_T570[n, x0, x1, g, gco : _T673, _T674, _T675, _T676, _T677] = +(X_T569[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T23[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T570    X_T569     X_T23  
//        g       576         1         1         1  
//       k0         3         0      8640      1728  
//       k1         3         0       576       576  
//       x0         7      4032     17280         0  
//       x1         7       576      1152         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 576, 3, 3, 7, 7 }
// Out stride: { 1, 0, 0, 4032, 576 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8640, 576, 17280, 1152 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1728, 576, 0, 0 }
// Tile size: { 32, 3, 3, 7, 7 }
// Contraction output var shape: fp32(1, 7, 7, 576, 1):(28224, 4032, 576, 1, 1):110.25 KiB
// Computed true ops: 508032
// Computed work groups: 18
// Computed inner loops: 1
// Computed shared mem: 29952
// Computed out regs: 8192
// Computed mem read: 29952
// Computed mem write: 6272
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4608, 1, 1
__kernel void kernel_c43_sdk_153(__global float* restrict  X_T570, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7200];
  __local float in2_shared[288];
  int g_gid = (get_group_id(0) * 32);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((g_gid + (k1_gid * 576)) + (k0_gid * 8640));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 29; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 28) || (k1_x1_k0_x0_tid < 1));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((225 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (576 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)129599)];
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
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int g_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
          {
            int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
            int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)(x1_cond && x0_cond));
              float val1 = in1_shared[(((((225 * g_tid) + k1_lid) + (2 * x1)) + (15 * k0_lid)) + (30 * x0))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              int agg_idx = (x1_lid + (x0_lid * 2));
              float agg_rhs = mad(val2, val1, agg[agg_idx]);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(x1_cond && x0_cond));
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
    int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T570 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = (((g_gid + g_tid) + (4032 * x0)) + (576 * x1));
          X_T570[gout_idx] = LX_T570;
        }
      }
    }
  }
}
