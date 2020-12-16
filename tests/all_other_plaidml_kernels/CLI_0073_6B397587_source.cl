#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 14 1
// lid: 256 1 1
// Original:
// X_T70[n, x0, x1, g, gco : _T54, _T55, _T56, _T57, _T58] = +(X_T69[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T67[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T70[n, x0, x1, g, gco : _T54, _T55, _T56, _T57, _T58] = +(X_T69[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T67[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= g < 32, 0 <= g + gci < 32, 0 <= g < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 117, 0 <= k1 + 2*x1 < 117, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= g < 32, 0 <= g + gci < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 117, 0 <= k1 + 2*x1 < 117 }
// Defracted:
// X_T70[n, x0, x1, g, gco : _T54, _T55, _T56, _T57, _T58] = +(X_T69[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T67[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T70     X_T69     X_T67  
//        g        32         1         1         1  
//       k0         7         0      3744       224  
//       k1         7         0        32        32  
//       x0        56      1792      7488         0  
//       x1        56        32        64         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 32, 7, 7, 56, 56 }
// Out stride: { 1, 0, 0, 1792, 32 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3744, 32, 7488, 64 }
// Input 2 offset: 0
// Input 2 stride: { 1, 224, 32, 0, 0 }
// Tile size: { 32, 4, 7, 4, 8 }
// Contraction output var shape: fp32(1, 56, 56, 32, 1):(100352, 1792, 32, 1, 1):392 KiB
// Computed true ops: 9834496
// Computed work groups: 98
// Computed inner loops: 2
// Computed shared mem: 30504
// Computed out regs: 4096
// Computed mem read: 30464
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 14, 1
__kernel void kernel_c42_sdk_11(__global float* restrict  X_T70, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[6730];
  __local float in2_shared[896];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 8; k0_gid += 4)
  {
    for (int k1_gid = 0; k1_gid < 7; k1_gid += 7)
    {
      {
        int gbase = ((((k1_gid * 32) + (x1_gid * 64)) + (k0_gid * 3744)) + (x0_gid * 7488));
        int g_k1_x1_tid = (tid % 256);
        for (int g_k1_x1_lid = 0; g_k1_x1_lid < 3; g_k1_x1_lid += 1)
        {
          int g_k1_x1_cond = ((g_k1_x1_lid < 2) || (g_k1_x1_tid < 160));
          if (g_k1_x1_cond)
          {
            int g_k1_x1 = ((256 * g_k1_x1_lid) + g_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 10; k0_x0_lid += 1)
            {
              int lidx = (g_k1_x1 + (673 * k0_x0_lid));
              int gidx = ((gbase + g_k1_x1) + (3744 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)438047)];
            }
          }
        }
      }
      {
        int gbase = ((k1_gid * 32) + (k0_gid * 224));
        int g_k1_k0_tid = (tid % 256);
        for (int g_k1_k0_lid = 0; g_k1_k0_lid < 4; g_k1_k0_lid += 1)
        {
          int g_k1_k0_cond = ((g_k1_k0_lid < 3) || (g_k1_k0_tid < 128));
          if (g_k1_k0_cond)
          {
            int g_k1_k0 = ((256 * g_k1_k0_lid) + g_k1_k0_tid);
            int gidx = (gbase + g_k1_k0);
            in2_shared[g_k1_k0] = in2[clamp((int)gidx, (int)0, (int)1567)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 4; k0_lid += 1)
      {
        int k0_cond = ((k0_lid < 3) || (k0_gid != 4));
        if (k0_cond)
        {
          for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
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
                float val1 = in1_shared[((((g_tid + (32 * k1_lid)) + (64 * x1)) + (673 * k0_lid)) + (1346 * x0))];
                float val2 = in2_shared[((g_tid + (32 * k1_lid)) + (224 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 2));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = agg_rhs;
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
    int x1 = ((4 * x1_lid) + x1_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T70 = agg[(x1_lid + (x0_lid * 2))];
      int gout_idx = ((g_tid + (1792 * (x0_gid + x0))) + (32 * (x1_gid + x1)));
      X_T70[gout_idx] = LX_T70;
    }
  }
}
