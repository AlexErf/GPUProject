#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T163[n, x0, x1, co : _T158, _T159, _T160, _T161] = +(X_T162[n, k0 + x0, k1 + x1, ci] * X_I_100[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T163[n, x0, x1, co : _T158, _T159, _T160, _T161] = +(X_T162[n, k0 + x0, k1 + x1, ci] * X_I_100[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 24, 0 <= co < 24, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= ci < 144, 0 <= ci < 144, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 24, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= ci < 144 }
// Defracted:
// X_T163[n, x0, x1, co : _T158, _T159, _T160, _T161] = +(X_T162[n, k0 + x0, k1 + x1, ci] * X_I_100[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T163    X_T162   X_I_100  
//       ci       144         0         1        24  
//       co        24         1         0         1  
//       x0        56      1344      8064         0  
//       x1        56        24       144         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 144, 24, 56, 56 }
// Out stride: { 0, 1, 1344, 24 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 8064, 144 }
// Input 2 offset: 0
// Input 2 stride: { 24, 1, 0, 0 }
// Elementwise input X_I_99 shape: fp32(24):(1):96 bytes
// Elementwise input X_I_98 shape: fp32(24):(1):96 bytes
// Elementwise op: [[pid(Sub)]] X_T164 = sub(X_T163, X_I_99)
// Elementwise op: [[pid(Mul)]] X_T165 = mul(X_T164, X_I_98)
// Tile size: { 32, 24, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 24):(75264, 1344, 24, 1):294 KiB
// Computed true ops: 43352064
// Computed work groups: 28
// Computed inner loops: 5
// Computed shared mem: 17672
// Computed out regs: 14336
// Computed mem read: 18304
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c43_sdk_37(__global float* restrict  X_T165, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_99, __global const float* restrict  X_I_98)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3650];
  __local float in2_shared[768];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 160; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 144));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 2);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        int lidx = (((57 * ci_tid) + (1825 * x1_tid)) + x0);
        int gidx = (((gbase + ci_tid) + (144 * x1_tid)) + (8064 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)451583)];
      }
    }
    {
      int gbase = (ci_gid * 24);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 3; co_ci_lid += 1)
      {
        int co_ci = ((256 * co_ci_lid) + co_ci_tid);
        int gidx = (gbase + co_ci);
        in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)3455)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int ci_cond = ((ci_lid < 16) || (ci_gid != 128));
      if (ci_cond)
      {
        int co_tid = (tid % 32);
        int x1_tid = ((tid / 32) % 2);
        int x0_tid = ((tid / 64) % 4);
        int co_cond = (co_tid < 24);
        int co = select((int)0, (int)co_tid, (int)co_cond);
        for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
        {
          int x0 = ((4 * x0_lid) + x0_tid);
          float val1 = in1_shared[(((57 * ci_lid) + (1825 * x1_tid)) + x0)];
          float val2 = in2_shared[(co + (24 * ci_lid))];
          float agg_rhs = mad(val2, val1, agg[x0_lid]);
          agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)co_cond);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int co_cond = (co_tid < 24);
  if (co_cond)
  {
    for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T163 = agg[x0_lid];
      int gout_idx = ((co_tid + (1344 * x0)) + (24 * (x1_gid + x1_tid)));
      float LX_I_99 = X_I_99[co_tid];
      float LX_I_98 = X_I_98[co_tid];
      float LX_T164 = (LX_T163 - LX_I_99);
      float LX_T165 = (LX_T164 * LX_I_98);
      X_T165[gout_idx] = LX_T165;
    }
  }
}
