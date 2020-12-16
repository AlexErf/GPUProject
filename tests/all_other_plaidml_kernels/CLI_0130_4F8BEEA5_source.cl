#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T507[n, x0, x1, co : _T592, _T593, _T594, _T595] = +(X_T506[n, k0 + x0, k1 + x1, ci] * X_I_199[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T507[n, x0, x1, co : _T592, _T593, _T594, _T595] = +(X_T506[n, k0 + x0, k1 + x1, ci] * X_I_199[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 96, 0 <= co < 96, 0 <= ci < 576, 0 <= ci < 576, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 96, 0 <= ci < 576 }
// Defracted:
// X_T507[n, x0, x1, co : _T592, _T593, _T594, _T595] = +(X_T506[n, k0 + x0, k1 + x1, ci] * X_I_199[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T507    X_T506   X_I_199  
//       ci       576         0         1        96  
//       co        96         1         0         1  
//       x0        14      1344      8064         0  
//       x1        14        96       576         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 576, 96, 14, 14 }
// Out stride: { 0, 1, 1344, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 8064, 576 }
// Input 2 offset: 0
// Input 2 stride: { 96, 1, 0, 0 }
// Elementwise input X_I_198 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_197 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Sub)]] X_T508 = sub(X_T507, X_I_198)
// Elementwise op: [[pid(Mul)]] X_T509 = mul(X_T508, X_I_197)
// Tile size: { 64, 32, 14, 2 }
// Contraction output var shape: fp32(1, 14, 14, 96):(18816, 1344, 96, 1):73.5 KiB
// Computed true ops: 43352064
// Computed work groups: 21
// Computed inner loops: 9
// Computed shared mem: 15608
// Computed out regs: 4096
// Computed mem read: 15584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c43_sdk_135(__global float* restrict  X_T509, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_198, __global const float* restrict  X_I_197)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1822];
  __local float in2_shared[2080];
  int co_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int ci_gid = 0; ci_gid < 576; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 576));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 2);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0));
        int gidx = (((gbase + ci_tid) + (576 * x1_tid)) + (8064 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)112895)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 96));
      int co_tid = (tid % 32);
      int ci_tid = ((tid / 32) % 8);
      for (int ci_lid = 0; ci_lid < 8; ci_lid += 1)
      {
        int ci = ((8 * ci_lid) + ci_tid);
        int lidx = ((65 * co_tid) + ci);
        int gidx = ((gbase + co_tid) + (96 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)55295)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 2);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
        int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
        float val1 = in1_shared[((ci_lid + (911 * x1_tid)) + (65 * x0))];
        float val2 = in2_shared[((65 * co_tid) + ci_lid)];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
    if (x0_cond)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T507 = agg[x0_lid];
      int gout_idx = (((co_gid + co_tid) + (1344 * x0)) + (96 * (x1_gid + x1_tid)));
      float LX_I_198 = X_I_198[(co_gid + co_tid)];
      float LX_I_197 = X_I_197[(co_gid + co_tid)];
      float LX_T508 = (LX_T507 - LX_I_198);
      float LX_T509 = (LX_T508 * LX_I_197);
      X_T509[gout_idx] = LX_T509;
    }
  }
}
