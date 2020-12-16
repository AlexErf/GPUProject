#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 1 1
// lid: 256 1 1
// Original:
// X_T46[n, x0, x1, co : _T24, _T25, _T26, _T27] = +(X_T45[n, k0 + x0, k1 + x1, ci] * X_I_41[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T46[n, x0, x1, co : _T24, _T25, _T26, _T27] = +(X_T45[n, k0 + x0, k1 + x1, ci] * X_I_41[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= co < 11, 0 <= ci < 32, 0 <= ci < 32, 0 <= x0 < 111, 0 <= x1 < 111, 0 <= k0 + x0 < 111, 0 <= k1 + x1 < 111, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= ci < 32, 0 <= x0 < 111, 0 <= x1 < 111, 0 <= k0 + x0 < 111, 0 <= k1 + x1 < 111 }
// Defracted:
// X_T46[n, x0, x1, co : _T24, _T25, _T26, _T27] = +(X_T45[n, k0 + x0, k1 + x1, ci] * X_I_41[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T46     X_T45    X_I_41  
//       ci        32         0         1        11  
//       co        11         1         0         1  
//       x0       111      1221      3552         0  
//       x1       111        11        32         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 32, 11, 111, 111 }
// Out stride: { 0, 1, 1221, 11 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3552, 32 }
// Input 2 offset: 0
// Input 2 stride: { 11, 1, 0, 0 }
// Elementwise input X_I_40 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_39 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(Sub)]] X_T47 = sub(X_T46, X_I_40)
// Elementwise op: [[pid(Mul)]] X_T48 = mul(X_T47, X_I_39)
// Tile size: { 32, 11, 111, 2 }
// Contraction output var shape: fp32(1, 111, 111, 11):(135531, 1221, 11, 1):529.418 KiB
// Computed true ops: 17347968
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 29824
// Computed out regs: 14336
// Computed mem read: 31600
// Computed mem write: 28416
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 1, 1
__kernel void kernel_c42_sdk_3(__global float* restrict  X_T48, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_40, __global const float* restrict  X_I_39)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7104];
  __local float in2_shared[352];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 32; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 32));
      int ci_x1_tid = (tid % 64);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 28; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 27) || (x0_tid < 3));
        if (x0_cond)
        {
          int x0 = ((4 * x0_lid) + x0_tid);
          int lidx = ((111 * ci_x1_tid) + x0);
          int gidx = ((gbase + ci_x1_tid) + (3552 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)394271)];
        }
      }
    }
    {
      int gbase = (ci_gid * 11);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 2; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 1) || (co_ci_tid < 96));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)351)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 16);
      int x1_tid = ((tid / 16) % 2);
      int x0_tid = ((tid / 32) % 8);
      int co_cond = (co_tid < 11);
      int co = select((int)0, (int)co_tid, (int)co_cond);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 13) || (x0_tid < 7));
        int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)(co_cond && x0_cond));
        float val1 = in1_shared[(((111 * ci_lid) + (3552 * x1_tid)) + x0)];
        float val2 = in2_shared[(co + (11 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(co_cond && x0_cond));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 2);
  int x0_tid = ((tid / 32) % 8);
  int x1_cond = ((x1_gid != 110) || (x1_tid < 1));
  if (x1_cond)
  {
    int co_cond = (co_tid < 11);
    if (co_cond)
    {
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 13) || (x0_tid < 7));
        if (x0_cond)
        {
          int x0 = ((8 * x0_lid) + x0_tid);
          float LX_T46 = agg[x0_lid];
          int gout_idx = ((co_tid + (1221 * x0)) + (11 * (x1_gid + x1_tid)));
          float LX_I_40 = X_I_40[co_tid];
          float LX_I_39 = X_I_39[co_tid];
          float LX_T47 = (LX_T46 - LX_I_40);
          float LX_T48 = (LX_T47 * LX_I_39);
          X_T48[gout_idx] = LX_T48;
        }
      }
    }
  }
}
