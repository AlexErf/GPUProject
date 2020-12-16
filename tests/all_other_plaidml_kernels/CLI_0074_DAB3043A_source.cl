#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T72[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T71[n, k0 + x0, k1 + x1, ci] * X_I_57[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T72[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T71[n, k0 + x0, k1 + x1, ci] * X_I_57[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= co < 11, 0 <= ci < 32, 0 <= ci < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 11, 0 <= ci < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56 }
// Defracted:
// X_T72[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T71[n, k0 + x0, k1 + x1, ci] * X_I_57[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T72     X_T71    X_I_57  
//       ci        32         0         1        11  
//       co        11         1         0         1  
//       x0        56       616      1792         0  
//       x1        56        11        32         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 32, 11, 56, 56 }
// Out stride: { 0, 1, 616, 11 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 1792, 32 }
// Input 2 offset: 0
// Input 2 stride: { 11, 1, 0, 0 }
// Elementwise input X_I_56 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_55 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(Sub)]] X_T73 = sub(X_T72, X_I_56)
// Elementwise op: [[pid(Mul)]] X_T74 = mul(X_T73, X_I_55)
// Tile size: { 32, 11, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 4415488
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 15968
// Computed out regs: 7168
// Computed mem read: 16640
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_12(__global float* restrict  X_T74, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_56, __global const float* restrict  X_I_55)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3640];
  __local float in2_shared[352];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 32; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 32));
      int ci_x1_tid = (tid % 64);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        int lidx = (ci_x1_tid + (65 * x0));
        int gidx = ((gbase + ci_x1_tid) + (1792 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
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
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((8 * x0_lid) + x0_tid);
        float val1 = in1_shared[((ci_lid + (32 * x1_tid)) + (65 * x0))];
        float val2 = in2_shared[(co + (11 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)co_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 2);
  int x0_tid = ((tid / 32) % 8);
  int co_cond = (co_tid < 11);
  if (co_cond)
  {
    for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      float LX_T72 = agg[x0_lid];
      int gout_idx = ((co_tid + (616 * x0)) + (11 * (x1_gid + x1_tid)));
      float LX_I_56 = X_I_56[co_tid];
      float LX_I_55 = X_I_55[co_tid];
      float LX_T73 = (LX_T72 - LX_I_56);
      float LX_T74 = (LX_T73 * LX_I_55);
      X_T74[gout_idx] = LX_T74;
    }
  }
}
