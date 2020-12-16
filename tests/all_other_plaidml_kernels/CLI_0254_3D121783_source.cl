#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 1
// lid: 256 1 1
// Original:
// X_T2988[n, x0, x1, g, gco : _T4754, _T4755, _T4756, _T4757, _T4758] = +(X_T2987[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T2985[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T2988[n, x0, x1, g, gco : _T4754, _T4755, _T4756, _T4757, _T4758] = +(X_T2987[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T2985[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -1 + k0 + x0 < 11, 0 <= -1 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672, 0 <= g < 672, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -1 + k0 + x0 < 11, 0 <= -1 + k1 + x1 < 11, 0 <= g < 672, 0 <= g + gci < 672 }
// Defracted:
// X_T2988[n, x0, x1, g, gco : _T4754, _T4755, _T4756, _T4757, _T4758] = +(X_T2987[n, -1 + k0 + x0, -1 + k1 + x1, g + gci] * X_T2985[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2988   X_T2987   X_T2985  
//        g       672         1         1         1  
//       k0         3         0      7392      2016  
//       k1         3         0       672       672  
//       x0        11      7392      7392         0  
//       x1        11       672       672         0  
//      off                   0     -8064         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 11
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 11
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 672, 3, 3, 11, 11 }
// Out stride: { 1, 0, 0, 7392, 672 }
// Input 1 offset: -8064
// Input 1 stride: { 1, 7392, 672, 7392, 672 }
// Input 2 offset: 0
// Input 2 stride: { 1, 2016, 672, 0, 0 }
// Tile size: { 32, 3, 3, 4, 11 }
// Contraction output var shape: fp32(1, 11, 11, 672, 1):(81312, 7392, 672, 1, 1):317.625 KiB
// Computed true ops: 1463616
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 9984
// Computed out regs: 6144
// Computed mem read: 9856
// Computed mem write: 5632
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 1
__kernel void kernel_c42_sdk_1156(__global float* restrict  X_T2988, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -8064);
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2208];
  __local float in2_shared[288];
  int g_gid = (get_group_id(1) * 32);
  int x0_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 672)) + (k0_gid * 7392)) + (x0_gid * 7392));
        int g_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 9; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 8) || (k1_x1_k0_x0_tid < 4));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((69 * g_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + g_tid) + (672 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)8064, (int)89375)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 672)) + (k0_gid * 2016));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 2; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 1) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((9 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (672 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)6047)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 4) - 1)) <= 11)) && ((-1 * k1_gid) <= -1)) && ((((k1_gid + 3) - 1) + 10) <= 11)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((69 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0_tid))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
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
            for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
            {
              int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
              int x1 = select((int)0, (int)((2 * x1_lid) + x1_tid), (int)x1_cond);
              float val1 = in1_shared[(((((69 * g_tid) + k1_lid) + x1) + (11 * k0_lid)) + (11 * x0_tid))];
              float val2 = in2_shared[(((9 * g_tid) + k1_lid) + (3 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x1_lid]);
              agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)(x1_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0_tid))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0_tid)) <= 11)) && (((-1 * (k1_gid + k1_lid)) + (-1 * x1)) <= -1)) && (((k1_gid + k1_lid) + x1) <= 11))));
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
  int x0_cond = ((x0_gid != 8) || (x0_tid < 3));
  if (x0_cond)
  {
    for (int x1_lid = 0; x1_lid < 6; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 5) || (x1_tid < 1));
      if (x1_cond)
      {
        int x1 = ((2 * x1_lid) + x1_tid);
        float LX_T2988 = agg[x1_lid];
        int gout_idx = (((g_gid + g_tid) + (7392 * (x0_gid + x0_tid))) + (672 * x1));
        if (((gout_idx >= 0) && (gout_idx < 81312)))
        {
          X_T2988[gout_idx] = LX_T2988;
        }
      }
    }
  }
}
