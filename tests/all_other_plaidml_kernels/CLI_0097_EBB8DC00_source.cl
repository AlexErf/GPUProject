#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 7 1
// lid: 256 1 1
// Original:
// X_T188[n, x0, x1, g, gco : _T191, _T192, _T193, _T194, _T195] = +(X_T187[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T39[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T188[n, x0, x1, g, gco : _T191, _T192, _T193, _T194, _T195] = +(X_T187[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T39[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57, 0 <= g < 144, 0 <= g + gci < 144, 0 <= g < 144, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57, 0 <= g < 144, 0 <= g + gci < 144 }
// Defracted:
// X_T188[n, x0, x1, g, gco : _T191, _T192, _T193, _T194, _T195] = +(X_T187[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T39[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T188    X_T187     X_T39  
//        g       144         1         1         1  
//       k0         3         0      8208       432  
//       k1         3         0       144       144  
//       x0        28      4032     16416         0  
//       x1        28       144       288         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 144, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 4032, 144 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8208, 144, 16416, 288 }
// Input 2 offset: 0
// Input 2 stride: { 1, 432, 144, 0, 0 }
// Tile size: { 32, 1, 3, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 144, 1):(112896, 4032, 144, 1, 1):441 KiB
// Computed true ops: 2032128
// Computed work groups: 35
// Computed inner loops: 3
// Computed shared mem: 29584
// Computed out regs: 14336
// Computed mem read: 29568
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 7, 1
__kernel void kernel_c43_sdk_44(__global float* restrict  X_T188, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7300];
  __local float in2_shared[96];
  int g_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 144)) + (k0_gid * 8208)) + (x0_gid * 16416));
        int g_tid = (tid % 32);
        int x0_tid = ((tid / 32) % 4);
        int k1_x1_k0_tid = ((tid / 128) % 2);
        for (int k1_x1_k0_lid = 0; k1_x1_k0_lid < 29; k1_x1_k0_lid += 1)
        {
          int k1_x1_k0_cond = ((k1_x1_k0_lid < 28) || (k1_x1_k0_tid < 1));
          if (k1_x1_k0_cond)
          {
            int k1_x1_k0 = ((2 * k1_x1_k0_lid) + k1_x1_k0_tid);
            int lidx = (((57 * g_tid) + (1825 * x0_tid)) + k1_x1_k0);
            int gidx = (((gbase + g_tid) + (16416 * x0_tid)) + (144 * k1_x1_k0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)467855)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 144)) + (k0_gid * 432));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 4);
        int k1_k0_cond = (k1_k0_tid < 3);
        if (k1_k0_cond)
        {
          if ((tid < 128))
          {
            int lidx = ((3 * g_tid) + k1_k0_tid);
            int gidx = ((gbase + g_tid) + (144 * k1_k0_tid));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)1295)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
      {
        int g_tid = (tid % 32);
        int x1_tid = ((tid / 32) % 4);
        int x0_tid = ((tid / 128) % 2);
        for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((((57 * g_tid) + k1_lid) + (2 * x1)) + (1825 * x0))];
            float val2 = in2_shared[((3 * g_tid) + k1_lid)];
            int agg_idx = (x1_lid + (x0_lid * 7));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = agg_rhs;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int g_cond = ((g_gid != 128) || (g_tid < 16));
  if (g_cond)
  {
    for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T188 = agg[(x1_lid + (x0_lid * 7))];
        int gout_idx = (((g_gid + g_tid) + (4032 * (x0_gid + x0))) + (144 * x1));
        X_T188[gout_idx] = LX_T188;
      }
    }
  }
}
