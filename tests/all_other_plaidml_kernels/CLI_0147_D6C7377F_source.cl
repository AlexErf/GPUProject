#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 1
// lid: 256 1 1
// Original:
// X_T485[n, x0, x1, co : _T730, _T731, _T732, _T733] = +(X_T484[n, k0 + x0, k1 + x1, ci] * X_I_194[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T485[n, x0, x1, co : _T730, _T731, _T732, _T733] = +(X_T484[n, k0 + x0, k1 + x1, ci] * X_I_194[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= co < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= ci < 44, 0 <= ci < 44, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= ci < 44 }
// Defracted:
// X_T485[n, x0, x1, co : _T730, _T731, _T732, _T733] = +(X_T484[n, k0 + x0, k1 + x1, ci] * X_I_194[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T485    X_T484   X_I_194  
//       ci        44         0         1        22  
//       co        22         1         0         1  
//       x0        28       616      1232         0  
//       x1        28        22        44         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 44, 22, 28, 28 }
// Out stride: { 0, 1, 616, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 1232, 44 }
// Input 2 offset: 0
// Input 2 stride: { 22, 1, 0, 0 }
// Tile size: { 44, 22, 4, 16 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 1517824
// Computed work groups: 14
// Computed inner loops: 1
// Computed shared mem: 15152
// Computed out regs: 8192
// Computed mem read: 15104
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 1
__kernel void kernel_c42_sdk_169(__global float* restrict  X_T485, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2820];
  __local float in2_shared[968];
  int x1_gid = (get_group_id(1) * 16);
  int x0_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 44; ci_gid += 44)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 44)) + (x0_gid * 1232));
      int ci_x1_tid = (tid % 256);
      for (int ci_x1_lid = 0; ci_x1_lid < 3; ci_x1_lid += 1)
      {
        int ci_x1_cond = ((ci_x1_lid < 2) || (ci_x1_tid < 192));
        if (ci_x1_cond)
        {
          int ci_x1 = ((256 * ci_x1_lid) + ci_x1_tid);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int lidx = (ci_x1 + (705 * x0_lid));
            int gidx = ((gbase + ci_x1) + (1232 * x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)34495)];
          }
        }
      }
    }
    {
      int gbase = (ci_gid * 22);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 4; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 3) || (co_ci_tid < 200));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)967)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 44; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        int co_cond = (co_tid < 22);
        int co = select((int)0, (int)co_tid, (int)co_cond);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (44 * x1)) + (705 * x0))];
          float val2 = in2_shared[(co + (22 * ci_lid))];
          int agg_idx = (x1_lid + (x0_lid * 4));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)co_cond);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 3) || (x1_gid != 16));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      int co_cond = (co_tid < 22);
      if (co_cond)
      {
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T485 = agg[(x1_lid + (x0_lid * 4))];
          int gout_idx = ((co_tid + (616 * (x0_gid + x0))) + (22 * (x1_gid + x1)));
          X_T485[gout_idx] = LX_T485;
        }
      }
    }
  }
}
