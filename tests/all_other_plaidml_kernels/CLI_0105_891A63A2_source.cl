#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Original:
// X_T229[n, x0, x1, g, gco : _T238, _T239, _T240, _T241, _T242] = +(X_T228[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T215[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T229[n, x0, x1, g, gco : _T238, _T239, _T240, _T241, _T242] = +(X_T228[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T215[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28, 0 <= g < 192, 0 <= g + gci < 192, 0 <= g < 192, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28, 0 <= g < 192, 0 <= g + gci < 192 }
// Defracted:
// X_T229[n, x0, x1, g, gco : _T238, _T239, _T240, _T241, _T242] = +(X_T228[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T215[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T229    X_T228    X_T215  
//        g       192         1         1         1  
//       k0         3         0      5376       576  
//       k1         3         0       192       192  
//       x0        28      5376      5376         0  
//       x1        28       192       192         0  
//      off                   0     -5568         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 28
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 28
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 192, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 5376, 192 }
// Input 1 offset: -5568
// Input 1 stride: { 1, 5376, 192, 5376, 192 }
// Input 2 offset: 0
// Input 2 stride: { 1, 576, 192, 0, 0 }
// Tile size: { 32, 3, 3, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 192, 1):(150528, 5376, 192, 1, 1):588 KiB
// Computed true ops: 2709504
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 23040
// Computed out regs: 14336
// Computed mem read: 22912
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c43_sdk_55(__global float* restrict  X_T229, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -5568);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[5472];
  __local float in2_shared[288];
  int g_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 192)) + (k0_gid * 5376)) + (x0_gid * 5376));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 22; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 21) || (k1_x1_k0_x0_tid < 2));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((171 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (192 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)5568, (int)156095)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 192)) + (k0_gid * 576));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (192 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)1727)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 28)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 27) <= 28)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
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
                float val1 = in1_shared[(((((171 * g_tid) + k1_lid) + x1) + (28 * k0_lid)) + (28 * x0))];
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
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                float val1 = in1_shared[(((((171 * g_tid) + k1_lid) + x1) + (28 * k0_lid)) + (28 * x0))];
                float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
                int agg_idx = (x1_lid + (x0_lid * 7));
                float agg_rhs = mad(val2, val1, agg[agg_idx]);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 28)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 28)));
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
  for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
  {
    int x1 = ((4 * x1_lid) + x1_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T229 = agg[(x1_lid + (x0_lid * 7))];
      int gout_idx = (((g_gid + g_tid) + (5376 * (x0_gid + x0))) + (192 * x1));
      if (((gout_idx >= 0) && (gout_idx < 150528)))
      {
        X_T229[gout_idx] = LX_T229;
      }
    }
  }
}
