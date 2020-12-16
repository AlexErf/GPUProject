#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T238[n, x0, x1, co : _T325, _T326, _T327, _T328] = +(X_T237[n, k0 + x0, k1 + x1, ci] * X_I_37[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T238[n, x0, x1, co : _T325, _T326, _T327, _T328] = +(X_T237[n, k0 + x0, k1 + x1, ci] * X_I_37[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= co < 22, 0 <= ci < 44, 0 <= ci < 44, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= ci < 44, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56 }
// Defracted:
// X_T238[n, x0, x1, co : _T325, _T326, _T327, _T328] = +(X_T237[n, k0 + x0, k1 + x1, ci] * X_I_37[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T238    X_T237    X_I_37  
//       ci        44         0         1        22  
//       co        22         1         0         1  
//       x0        56      1232      2464         0  
//       x1        56        22        44         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 44, 22, 56, 56 }
// Out stride: { 0, 1, 1232, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 2464, 44 }
// Input 2 offset: 0
// Input 2 stride: { 22, 1, 0, 0 }
// Elementwise input X_I_36 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_35 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(Sub)]] X_T239 = sub(X_T238, X_I_36)
// Elementwise op: [[pid(Mul)]] X_T240 = mul(X_T239, X_I_35)
// Tile size: { 44, 22, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Computed true ops: 12142592
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 23808
// Computed out regs: 14336
// Computed mem read: 24448
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_74(__global float* restrict  X_T240, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_36, __global const float* restrict  X_I_35)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4984];
  __local float in2_shared[968];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 44; ci_gid += 44)
  {
    {
      int gbase = (ci_gid + (x1_gid * 44));
      int ci_x1_tid = (tid % 128);
      int x0_tid = ((tid / 128) % 2);
      int ci_x1_cond = (ci_x1_tid < 88);
      if (ci_x1_cond)
      {
        for (int x0_lid = 0; x0_lid < 28; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          int lidx = (ci_x1_tid + (89 * x0));
          int gidx = ((gbase + ci_x1_tid) + (2464 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)137983)];
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
      int x1_tid = ((tid / 32) % 2);
      int x0_tid = ((tid / 64) % 4);
      int co_cond = (co_tid < 22);
      int co = select((int)0, (int)co_tid, (int)co_cond);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float val1 = in1_shared[((ci_lid + (44 * x1_tid)) + (89 * x0))];
        float val2 = in2_shared[(co + (22 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)co_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int co_cond = (co_tid < 22);
  if (co_cond)
  {
    for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T238 = agg[x0_lid];
      int gout_idx = ((co_tid + (1232 * x0)) + (22 * (x1_gid + x1_tid)));
      float LX_I_36 = X_I_36[co_tid];
      float LX_I_35 = X_I_35[co_tid];
      float LX_T239 = (LX_T238 - LX_I_36);
      float LX_T240 = (LX_T239 * LX_I_35);
      X_T240[gout_idx] = LX_T240;
    }
  }
}
