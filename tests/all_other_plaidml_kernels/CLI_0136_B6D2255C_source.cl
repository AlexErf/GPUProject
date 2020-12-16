#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 1
// lid: 256 1 1
// Original:
// X_T432[n, x0, x1, g, gco : _T648, _T649, _T650, _T651, _T652] = +(X_T431[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T429[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T432[n, x0, x1, g, gco : _T648, _T649, _T650, _T651, _T652] = +(X_T431[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T429[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= g < 84, 0 <= g + gci < 84, 0 <= g < 84, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= -1 + k0 + x0 < 42, 0 <= -1 + k1 + x1 < 42, 0 <= g < 84, 0 <= g + gci < 84 }
// Defracted:
// X_T432[n, x0, x1, g, gco : _T648, _T649, _T650, _T651, _T652] = +(X_T431[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T429[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T432    X_T431    X_T429  
//        g        84         1         1         1  
//       k0         3         0      3528       252  
//       k1         3         0        84        84  
//       x0        42      3528      3528         0  
//       x1        42        84        84         0  
//      off                   0     -3612         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 42
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 42
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 84, 3, 3, 42, 42 }
// Out stride: { 1, 0, 0, 3528, 84 }
// Input 1 offset: -3612
// Input 1 stride: { 1, 3528, 84, 3528, 84 }
// Input 2 offset: 0
// Input 2 stride: { 1, 252, 84, 0, 0 }
// Tile size: { 32, 3, 3, 42, 2 }
// Contraction output var shape: fp32(1, 42, 42, 84, 1):(148176, 3528, 84, 1, 1):578.812 KiB
// Computed true ops: 2667168
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 24208
// Computed out regs: 11264
// Computed mem read: 23680
// Computed mem write: 10752
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 1
__kernel void kernel_c42_sdk_150(__global float* restrict  X_T432, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -3612);
  int tid = get_local_id(0);
  float agg[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[5764];
  __local float in2_shared[288];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 84)) + (x1_gid * 84)) + (k0_gid * 3528));
        int g_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 4);
        int k0_x0_tid = ((tid / 128) % 2);
        for (int k0_x0_lid = 0; k0_x0_lid < 22; k0_x0_lid += 1)
        {
          int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
          int lidx = (((45 * g_tid) + (1441 * k1_x1_tid)) + k0_x0);
          int gidx = (((gbase + g_tid) + (84 * k1_x1_tid)) + (3528 * k0_x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)3612, (int)151787)];
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 84)) + (k0_gid * 252));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (84 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)755)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 41) <= 42)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 2) - 1)) <= 42)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((45 * g_tid) + (1441 * k1_lid)) + (1441 * x1_tid)) + k0_lid) + x0)];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
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
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((45 * g_tid) + (1441 * k1_lid)) + (1441 * x1_tid)) + k0_lid) + x0)];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 42)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 42))));
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
  int g_cond = ((g_gid != 64) || (g_tid < 20));
  if (g_cond)
  {
    for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 10) || (x0_tid < 2));
      if (x0_cond)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T432 = agg[x0_lid];
        int gout_idx = (((g_gid + g_tid) + (3528 * x0)) + (84 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 148176)))
        {
          X_T432[gout_idx] = LX_T432;
        }
      }
    }
  }
}
