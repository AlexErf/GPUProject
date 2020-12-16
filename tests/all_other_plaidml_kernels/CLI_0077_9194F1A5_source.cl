#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T85[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T84[n, k0 + x0, k1 + x1, ci] * X_I_52[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T85[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T84[n, k0 + x0, k1 + x1, ci] * X_I_52[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= ci < 11, 0 <= ci < 11, 0 <= co < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= ci < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56 }
// Defracted:
// X_T85[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T84[n, k0 + x0, k1 + x1, ci] * X_I_52[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T85     X_T84    X_I_52  
//       ci        11         0         1        11  
//       co        11         1         0         1  
//       x0        56       616       616         0  
//       x1        56        11        11         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 11, 11, 56, 56 }
// Out stride: { 0, 1, 616, 11 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 616, 11 }
// Input 2 offset: 0
// Input 2 stride: { 11, 1, 0, 0 }
// Elementwise input X_I_51 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_50 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(Sub)]] X_T86 = sub(X_T85, X_I_51)
// Elementwise op: [[pid(Mul)]] X_T87 = mul(X_T86, X_I_50)
// Tile size: { 11, 11, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 1517824
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 3332
// Computed out regs: 4096
// Computed mem read: 3712
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_16(__global float* restrict  X_T87, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_51, __global const float* restrict  X_I_50)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[712];
  __local float in2_shared[121];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int ci_gid = 0; ci_gid < 11; ci_gid += 11)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 11)) + (x0_gid * 616));
      int ci_x1_tid = (tid % 128);
      int x0_tid = ((tid / 128) % 2);
      int ci_x1_cond = (ci_x1_tid < 88);
      if (ci_x1_cond)
      {
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          int lidx = (ci_x1_tid + (89 * x0));
          int gidx = ((gbase + ci_x1_tid) + (616 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)34495)];
        }
      }
    }
    {
      int gbase = (ci_gid * 11);
      int co_ci_tid = (tid % 128);
      int co_ci_cond = (co_ci_tid < 121);
      if (co_ci_cond)
      {
        if ((tid < 128))
        {
          int gidx = (gbase + co_ci_tid);
          in2_shared[co_ci_tid] = in2[clamp((int)gidx, (int)0, (int)120)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 11; ci_lid += 1)
    {
      int co_tid = (tid % 16);
      int x1_tid = ((tid / 16) % 4);
      int x0_tid = ((tid / 64) % 4);
      int co_cond = (co_tid < 11);
      int co = select((int)0, (int)co_tid, (int)co_cond);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((4 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (11 * x1)) + (89 * x0))];
          float val2 = in2_shared[(co + (11 * ci_lid))];
          int agg_idx = (x1_lid + (x0_lid * 2));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)co_cond);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 4);
  int x0_tid = ((tid / 64) % 4);
  int co_cond = (co_tid < 11);
  if (co_cond)
  {
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T85 = agg[(x1_lid + (x0_lid * 2))];
        int gout_idx = ((co_tid + (616 * (x0_gid + x0))) + (11 * (x1_gid + x1)));
        float LX_I_51 = X_I_51[co_tid];
        float LX_I_50 = X_I_50[co_tid];
        float LX_T86 = (LX_T85 - LX_I_51);
        float LX_T87 = (LX_T86 * LX_I_50);
        X_T87[gout_idx] = LX_T87;
      }
    }
  }
}
