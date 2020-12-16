#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 14
// lid: 256 1 1
// Original:
// X_T108[n, x0, x1, g, gco : _T93, _T94, _T95, _T96, _T97] = +(X_T107[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T44[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T108[n, x0, x1, g, gco : _T93, _T94, _T95, _T96, _T97] = +(X_T107[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T44[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= g < 96, 0 <= g + gci < 96, 0 <= g < 96, 0 <= k0 + 2*x0 < 113, 0 <= k1 + 2*x1 < 113, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= g < 96, 0 <= g + gci < 96, 0 <= k0 + 2*x0 < 113, 0 <= k1 + 2*x1 < 113 }
// Defracted:
// X_T108[n, x0, x1, g, gco : _T93, _T94, _T95, _T96, _T97] = +(X_T107[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T44[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T108    X_T107     X_T44  
//        g        96         1         1         1  
//       k0         3         0     10848       288  
//       k1         3         0        96        96  
//       x0        56      5376     21696         0  
//       x1        56        96       192         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 96, 3, 3, 56, 56 }
// Out stride: { 1, 0, 0, 5376, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 10848, 96, 21696, 192 }
// Input 2 offset: 0
// Input 2 stride: { 1, 288, 96, 0, 0 }
// Tile size: { 32, 3, 3, 4, 8 }
// Contraction output var shape: fp32(1, 56, 56, 96, 1):(301056, 5376, 96, 1, 1):1176 KiB
// Computed true ops: 5419008
// Computed work groups: 294
// Computed inner loops: 1
// Computed shared mem: 20736
// Computed out regs: 4096
// Computed mem read: 20736
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 14
__kernel void kernel_c43_sdk_22(__global float* restrict  X_T108, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[4896];
  __local float in2_shared[288];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 8);
  int x0_gid = (get_group_id(2) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((g_gid + (k1_gid * 96)) + (x1_gid * 192)) + (k0_gid * 10848)) + (x0_gid * 21696));
        int g_tid = (tid % 32);
        int k0_x0_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 2; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 1) || (k0_x0_tid < 1));
          if (k0_x0_cond)
          {
            int k0_x0 = ((8 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 17; k1_x1_lid += 1)
            {
              int lidx = (((153 * g_tid) + (17 * k0_x0)) + k1_x1_lid);
              int gidx = (((gbase + g_tid) + (10848 * k0_x0)) + (96 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1225823)];
            }
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 96)) + (k0_gid * 288));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (96 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)863)];
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
            int x1 = ((4 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              float val1 = in1_shared[(((((153 * g_tid) + k1_lid) + (2 * x1)) + (17 * k0_lid)) + (34 * x0))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              int agg_idx = (x1_lid + (x0_lid * 2));
              float agg_rhs = mad(val2, val1, agg[agg_idx]);
              agg[agg_idx] = agg_rhs;
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
    int x1 = ((4 * x1_lid) + x1_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T108 = agg[(x1_lid + (x0_lid * 2))];
      int gout_idx = (((g_gid + g_tid) + (5376 * (x0_gid + x0))) + (96 * (x1_gid + x1)));
      X_T108[gout_idx] = LX_T108;
    }
  }
}
