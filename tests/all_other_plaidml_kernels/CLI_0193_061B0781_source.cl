#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 11 1
// lid: 256 1 1
// Original:
// X_T1631[n, x0, x1, g, gco : _T2568, _T2569, _T2570, _T2571, _T2572] = +(X_T1630[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1614[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T1631[n, x0, x1, g, gco : _T2568, _T2569, _T2570, _T2571, _T2572] = +(X_T1630[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1614[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= -2 + k0 + x0 < 21, 0 <= -2 + k1 + x1 < 21, 0 <= g < 336, 0 <= g + gci < 336, 0 <= g < 336, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= -2 + k0 + x0 < 21, 0 <= -2 + k1 + x1 < 21, 0 <= g < 336, 0 <= g + gci < 336 }
// Defracted:
// X_T1631[n, x0, x1, g, gco : _T2568, _T2569, _T2570, _T2571, _T2572] = +(X_T1630[n, -2 + k0 + x0, -2 + k1 + x1, g + gci] * X_T1614[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1631   X_T1630   X_T1614  
//        g       336         1         1         1  
//       k0         5         0      7056      1680  
//       k1         5         0       336       336  
//       x0        21      7056      7056         0  
//       x1        21       336       336         0  
//      off                   0    -14784         0  
//      vec                   1         1         1  
// Constraint: (0,-1,0,-1,0) <= -2
// Constraint: (0,1,0,1,0) <= 22
// Constraint: (0,0,-1,0,-1) <= -2
// Constraint: (0,0,1,0,1) <= 22
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 336, 5, 5, 21, 21 }
// Out stride: { 1, 0, 0, 7056, 336 }
// Input 1 offset: -14784
// Input 1 stride: { 1, 7056, 336, 7056, 336 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1680, 336, 0, 0 }
// Tile size: { 32, 5, 5, 21, 4 }
// Contraction output var shape: fp32(1, 21, 21, 336, 1):(148176, 7056, 336, 1, 1):578.812 KiB
// Computed true ops: 7408800
// Computed work groups: 66
// Computed inner loops: 1
// Computed shared mem: 28832
// Computed out regs: 11264
// Computed mem read: 28800
// Computed mem write: 10752
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 11, 1
__kernel void kernel_c42_sdk_620(__global float* restrict  X_T1631, __global const float* restrict  in1, __global const float* restrict  in2)
{
  in1 = (in1 + -14784);
  int tid = get_local_id(0);
  float agg[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6408];
  __local float in2_shared[800];
  int g_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = (((g_gid + (k1_gid * 336)) + (x1_gid * 336)) + (k0_gid * 7056));
        int g_tid = (tid % 32);
        int k1_x1_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 25; k0_x0_lid += 1)
        {
          int lidx = (((25 * g_tid) + (801 * k1_x1_tid)) + k0_x0_lid);
          int gidx = (((gbase + g_tid) + (336 * k1_x1_tid)) + (7056 * k0_x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)14784, (int)162959)];
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 336)) + (k0_gid * 1680));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 8);
        for (int k1_k0_lid = 0; k1_k0_lid < 4; k1_k0_lid += 1)
        {
          int k1_k0_cond = ((k1_k0_lid < 3) || (k1_k0_tid < 1));
          if (k1_k0_cond)
          {
            int k1_k0 = ((8 * k1_k0_lid) + k1_k0_tid);
            int lidx = ((25 * g_tid) + k1_k0);
            int gidx = ((gbase + g_tid) + (336 * k1_k0));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)8399)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -2) && ((((k0_gid + 5) - 1) + 20) <= 22)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -2)) && ((((k1_gid + 5) - 1) + ((x1_gid + 4) - 1)) <= 22)))
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((25 * g_tid) + (801 * k1_lid)) + (801 * x1_tid)) + k0_lid) + x0)];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
          {
            int g_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
              int x0 = select((int)0, (int)((2 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((((25 * g_tid) + (801 * k1_lid)) + (801 * x1_tid)) + k0_lid) + x0)];
              float val2 = in2_shared[(((25 * g_tid) + k1_lid) + (5 * k0_lid))];
              float agg_rhs = mad(val2, val1, agg[x0_lid]);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -2) && (((k0_gid + k0_lid) + x0) <= 22)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -2)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 22))));
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
  int g_cond = ((g_gid != 320) || (g_tid < 16));
  if (g_cond)
  {
    int x1_cond = ((x1_gid != 20) || (x1_tid < 1));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 11; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 10) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T1631 = agg[x0_lid];
          int gout_idx = (((g_gid + g_tid) + (7056 * x0)) + (336 * (x1_gid + x1_tid)));
          if (((gout_idx >= 0) && (gout_idx < 148176)))
          {
            X_T1631[gout_idx] = LX_T1631;
          }
        }
      }
    }
  }
}
