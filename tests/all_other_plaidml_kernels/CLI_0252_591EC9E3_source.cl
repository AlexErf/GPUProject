#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 2
// lid: 256 1 1
// Original:
// X_T347[n, x0, x1, co : _T410, _T411, _T412, _T413] = +(X_T346[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T347[n, x0, x1, co : _T410, _T411, _T412, _T413] = +(X_T346[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 128, 0 <= co < 128, 0 <= ci < 224, 0 <= ci < 224, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 128, 0 <= ci < 224 }
// Defracted:
// X_T347[n, x0, x1, co : _T410, _T411, _T412, _T413] = +(X_T346[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T347    X_T346   X_I_118  
//       ci       224         0         1       128  
//       co       128         1         0         1  
//       x0        28      3584      6272         0  
//       x1        28       128       224         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 224, 128, 28, 28 }
// Out stride: { 0, 1, 3584, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 6272, 224 }
// Input 2 offset: 0
// Input 2 stride: { 128, 1, 0, 0 }
// Elementwise input X_I_117 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_116 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Sub)]] X_T348 = sub(X_T347, X_I_117)
// Elementwise op: [[pid(Mul)]] X_T349 = mul(X_T348, X_I_116)
// Tile size: { 32, 64, 4, 16 }
// Contraction output var shape: fp32(1, 28, 28, 128):(100352, 3584, 128, 1):392 KiB
// Computed true ops: 89915392
// Computed work groups: 28
// Computed inner loops: 7
// Computed shared mem: 16784
// Computed out regs: 16384
// Computed mem read: 17408
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 2
__kernel void kernel_c124_sdk_96(__global float* restrict  X_T349, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_117, __global const float* restrict  X_I_116)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2116];
  __local float in2_shared[2080];
  int co_gid = (get_group_id(1) * 64);
  int x1_gid = (get_group_id(2) * 16);
  int x0_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 224; ci_gid += 32)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 224)) + (x0_gid * 6272));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 8);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((8 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int lidx = ((ci_tid + (33 * x1)) + (529 * x0_lid));
          int gidx = (((gbase + ci_tid) + (224 * x1)) + (6272 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)175615)];
        }
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 128));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 8; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (128 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)28671)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int co_lid = 0; co_lid < 2; co_lid += 1)
        {
          int co = ((32 * co_lid) + co_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (33 * x1)) + (529 * x0))];
            float val2 = in2_shared[(co + (65 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 8));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = agg_rhs;
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
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T347 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 8))];
          int gout_idx = (((co_gid + co) + (3584 * (x0_gid + x0))) + (128 * (x1_gid + x1)));
          float LX_I_117 = X_I_117[(co_gid + co)];
          float LX_I_116 = X_I_116[(co_gid + co)];
          float LX_T348 = (LX_T347 - LX_I_117);
          float LX_T349 = (LX_T348 * LX_I_116);
          X_T349[gout_idx] = LX_T349;
        }
      }
    }
  }
}
