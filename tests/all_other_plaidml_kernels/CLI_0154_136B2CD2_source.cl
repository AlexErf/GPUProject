#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 1
// lid: 256 1 1
// Original:
// X_T526[n, x0, x1, co : _T807, _T808, _T809, _T810] = +(X_T525[n, k0 + x0, k1 + x1, ci] * X_I_205[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T526[n, x0, x1, co : _T807, _T808, _T809, _T810] = +(X_T525[n, k0 + x0, k1 + x1, ci] * X_I_205[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 44, 0 <= ci < 44, 0 <= ci < 44, 0 <= co < 44, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 44, 0 <= ci < 44 }
// Defracted:
// X_T526[n, x0, x1, co : _T807, _T808, _T809, _T810] = +(X_T525[n, k0 + x0, k1 + x1, ci] * X_I_205[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T526    X_T525   X_I_205  
//       ci        44         0         1        44  
//       co        44         1         0         1  
//       x0        28      1232      1232         0  
//       x1        28        44        44         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 44, 44, 28, 28 }
// Out stride: { 0, 1, 1232, 44 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 1232, 44 }
// Input 2 offset: 0
// Input 2 stride: { 44, 1, 0, 0 }
// Elementwise input X_I_204 shape: fp32(44):(1):176 bytes
// Elementwise input X_I_203 shape: fp32(44):(1):176 bytes
// Elementwise op: [[pid(Sub)]] X_T527 = sub(X_T526, X_I_204)
// Elementwise op: [[pid(Mul)]] X_T528 = mul(X_T527, X_I_203)
// Tile size: { 44, 44, 4, 16 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 6071296
// Computed work groups: 14
// Computed inner loops: 1
// Computed shared mem: 19024
// Computed out regs: 16384
// Computed mem read: 19968
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 1
__kernel void kernel_c42_sdk_188(__global float* restrict  X_T528, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_204, __global const float* restrict  X_I_203)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2820];
  __local float in2_shared[1936];
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
      int gbase = (ci_gid * 44);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 8; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 7) || (co_ci_tid < 144));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)1935)];
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
        for (int co_lid = 0; co_lid < 2; co_lid += 1)
        {
          int co_cond = ((co_lid < 1) || (co_tid < 12));
          int co = select((int)0, (int)((32 * co_lid) + co_tid), (int)co_cond);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (44 * x1)) + (705 * x0))];
            float val2 = in2_shared[(co + (44 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 8));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)co_cond);
          }
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
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co_cond = ((co_lid < 1) || (co_tid < 12));
        if (co_cond)
        {
          int co = ((32 * co_lid) + co_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float LX_T526 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 8))];
            int gout_idx = ((co + (1232 * (x0_gid + x0))) + (44 * (x1_gid + x1)));
            float LX_I_204 = X_I_204[co];
            float LX_I_203 = X_I_203[co];
            float LX_T527 = (LX_T526 - LX_I_204);
            float LX_T528 = (LX_T527 * LX_I_203);
            X_T528[gout_idx] = LX_T528;
          }
        }
      }
    }
  }
}
