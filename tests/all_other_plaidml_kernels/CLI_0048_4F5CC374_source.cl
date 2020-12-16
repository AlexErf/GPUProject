#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Original:
// X_T101[n, x0, x1, co : _T70, _T71, _T72, _T73] = +(X_T100[n, k0 + x0, k1 + x1, ci] * X_I_102[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T101[n, x0, x1, co : _T70, _T71, _T72, _T73] = +(X_T100[n, k0 + x0, k1 + x1, ci] * X_I_102[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 32, 0 <= ci < 32, 0 <= co < 64, 0 <= co < 64, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 32, 0 <= co < 64, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112 }
// Defracted:
// X_T101[n, x0, x1, co : _T70, _T71, _T72, _T73] = +(X_T100[n, k0 + x0, k1 + x1, ci] * X_I_102[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T101    X_T100   X_I_102  
//       ci        32         0         1        64  
//       co        64         1         0         1  
//       x0       112      7168      3584         0  
//       x1       112        64        32         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 32, 64, 112, 112 }
// Out stride: { 0, 1, 7168, 64 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3584, 32 }
// Input 2 offset: 0
// Input 2 stride: { 64, 1, 0, 0 }
// Elementwise input X_I_101 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_100 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Sub)]] X_T102 = sub(X_T101, X_I_101)
// Elementwise op: [[pid(Mul)]] X_T103 = mul(X_T102, X_I_100)
// Tile size: { 32, 64, 16, 4 }
// Contraction output var shape: fp32(1, 112, 112, 64):(802816, 7168, 64, 1):3136 KiB
// Computed true ops: 102760448
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 16448
// Computed out regs: 16384
// Computed mem read: 17408
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c25_sdk_22(__global float* restrict  X_T103, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_101, __global const float* restrict  X_I_100)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2064];
  __local float in2_shared[2048];
  int x1_gid = (get_group_id(1) * 4);
  int x0_gid = (get_group_id(0) * 16);
  for (int ci_gid = 0; ci_gid < 32; ci_gid += 32)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 32)) + (x0_gid * 3584));
      int ci_x1_tid = (tid % 128);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        int lidx = (ci_x1_tid + (129 * x0));
        int gidx = ((gbase + ci_x1_tid) + (3584 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
      }
    }
    {
      int gbase = (ci_gid * 64);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 8; co_ci_lid += 1)
      {
        int co_ci = ((256 * co_ci_lid) + co_ci_tid);
        int gidx = (gbase + co_ci);
        in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)2047)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (32 * x1_tid)) + (129 * x0))];
          float val2 = in2_shared[(co + (64 * ci_lid))];
          int agg_idx = (co_lid + (x0_lid * 2));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = agg_rhs;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int co_lid = 0; co_lid < 2; co_lid += 1)
  {
    int co = ((32 * co_lid) + co_tid);
    for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T101 = agg[(co_lid + (x0_lid * 2))];
      int gout_idx = ((co + (7168 * (x0_gid + x0))) + (64 * (x1_gid + x1_tid)));
      float LX_I_101 = X_I_101[co];
      float LX_I_100 = X_I_100[co];
      float LX_T102 = (LX_T101 - LX_I_101);
      float LX_T103 = (LX_T102 * LX_I_100);
      X_T103[gout_idx] = LX_T103;
    }
  }
}
