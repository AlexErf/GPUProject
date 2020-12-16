#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T147[n, x0, x1, g, gco : _T173, _T174, _T175, _T176, _T177] = +(X_T146[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T143[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T147[n, x0, x1, g, gco : _T173, _T174, _T175, _T176, _T177] = +(X_T146[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T143[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 11, 0 <= g + gci < 11, 0 <= g < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 115, 0 <= k1 + 2*x1 < 115, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 11, 0 <= g + gci < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 115, 0 <= k1 + 2*x1 < 115 }
// Defracted:
// X_T147[n, x0, x1, g, gco : _T173, _T174, _T175, _T176, _T177] = +(X_T146[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T143[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T147    X_T146    X_T143  
//        g        11         1         1         1  
//       k0         5         0      1265        55  
//       k1         5         0        11        11  
//       x0        56       616      2530         0  
//       x1        56        11        22         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 11, 5, 5, 56, 56 }
// Out stride: { 1, 0, 0, 616, 11 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1265, 11, 2530, 22 }
// Input 2 offset: 0
// Input 2 stride: { 1, 55, 11, 0, 0 }
// Tile size: { 11, 5, 5, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 11, 1):(34496, 616, 11, 1, 1):134.75 KiB
// Computed true ops: 1724800
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 16984
// Computed out regs: 4096
// Computed mem read: 16896
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_39(__global float* restrict  X_T147, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3971];
  __local float in2_shared[275];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = ((((k1_gid * 11) + (x1_gid * 22)) + (k0_gid * 1265)) + (x0_gid * 2530));
        int g_k1_x1_tid = (tid % 256);
        int g_k1_x1_cond = (g_k1_x1_tid < 209);
        if (g_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 19; k0_x0_lid += 1)
          {
            int lidx = (g_k1_x1_tid + (209 * k0_x0_lid));
            int gidx = ((gbase + g_k1_x1_tid) + (1265 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)145474)];
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
      for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
        {
          int g_tid = (tid % 16);
          int x1_tid = ((tid / 16) % 4);
          int x0_tid = ((tid / 64) % 4);
          int g_cond = (g_tid < 11);
          int g = select((int)0, (int)g_tid, (int)g_cond);
          for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
          {
            int x1 = ((4 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((g + (11 * k1_lid)) + (22 * x1)) + (209 * k0_lid)) + (418 * x0))];
              float val2 = in2_shared[((g + (11 * k1_lid)) + (55 * k0_lid))];
              int agg_idx = (x1_lid + (x0_lid * 2));
              float agg_rhs = mad(val2, val1, agg[agg_idx]);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)g_cond);
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
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T147 = agg[(x1_lid + (x0_lid * 2))];
        int gout_idx = ((g_tid + (616 * (x0_gid + x0))) + (11 * (x1_gid + x1)));
        X_T147[gout_idx] = LX_T147;
      }
    }
  }
}
