#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 1
// lid: 256 1 1
// Original:
// X_T357[n, x0, x1, co : _T402, _T403, _T404, _T405] = +(X_T356[n, k0 + x0, k1 + x1, ci] * X_I_151[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T357[n, x0, x1, co : _T402, _T403, _T404, _T405] = +(X_T356[n, k0 + x0, k1 + x1, ci] * X_I_151[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 64, 0 <= co < 64, 0 <= ci < 384, 0 <= ci < 384, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 64, 0 <= ci < 384 }
// Defracted:
// X_T357[n, x0, x1, co : _T402, _T403, _T404, _T405] = +(X_T356[n, k0 + x0, k1 + x1, ci] * X_I_151[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T357    X_T356   X_I_151  
//       ci       384         0         1        64  
//       co        64         1         0         1  
//       x0        14       896      5376         0  
//       x1        14        64       384         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 384, 64, 14, 14 }
// Out stride: { 0, 1, 896, 64 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 5376, 384 }
// Input 2 offset: 0
// Input 2 stride: { 64, 1, 0, 0 }
// Elementwise input X_I_150 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_149 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Sub)]] X_T358 = sub(X_T357, X_I_150)
// Elementwise op: [[pid(Mul)]] X_T359 = mul(X_T358, X_I_149)
// Tile size: { 128, 32, 14, 2 }
// Contraction output var shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Computed true ops: 19267584
// Computed work groups: 14
// Computed inner loops: 3
// Computed shared mem: 30968
// Computed out regs: 4096
// Computed mem read: 30944
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 1
__kernel void kernel_c43_sdk_92(__global float* restrict  X_T359, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_150, __global const float* restrict  X_I_149)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3614];
  __local float in2_shared[4128];
  int co_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 384; ci_gid += 128)
  {
    {
      int gbase = (ci_gid + (x1_gid * 384));
      int ci_tid = (tid % 128);
      int x1_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (1807 * x1_tid)) + (129 * x0_lid));
        int gidx = (((gbase + ci_tid) + (384 * x1_tid)) + (5376 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)75263)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 64));
      int co_tid = (tid % 32);
      int ci_tid = ((tid / 32) % 8);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((8 * ci_lid) + ci_tid);
        int lidx = ((129 * co_tid) + ci);
        int gidx = ((gbase + co_tid) + (64 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)24575)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 128; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 2);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
        int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
        float val1 = in1_shared[((ci_lid + (1807 * x1_tid)) + (129 * x0))];
        float val2 = in2_shared[((129 * co_tid) + ci_lid)];
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
      float LX_T357 = agg[x0_lid];
      int gout_idx = (((co_gid + co_tid) + (896 * x0)) + (64 * (x1_gid + x1_tid)));
      float LX_I_150 = X_I_150[(co_gid + co_tid)];
      float LX_I_149 = X_I_149[(co_gid + co_tid)];
      float LX_T358 = (LX_T357 - LX_I_150);
      float LX_T359 = (LX_T358 * LX_I_149);
      X_T359[gout_idx] = LX_T359;
    }
  }
}
