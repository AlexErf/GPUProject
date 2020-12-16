#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 8 1
// lid: 256 1 1
// Original:
// X_T242[n, x0, x1, co : _T252, _T253, _T254, _T255] = +(X_T241[n, k0 + x0, k1 + x1, ci] * X_I_62[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T242[n, x0, x1, co : _T252, _T253, _T254, _T255] = +(X_T241[n, k0 + x0, k1 + x1, ci] * X_I_62[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= ci < 256, 0 <= ci < 256, 0 <= co < 512, 0 <= co < 512, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= ci < 256, 0 <= co < 512 }
// Defracted:
// X_T242[n, x0, x1, co : _T252, _T253, _T254, _T255] = +(X_T241[n, k0 + x0, k1 + x1, ci] * X_I_62[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T242    X_T241    X_I_62  
//       ci       256         0         1       512  
//       co       512         1         0         1  
//       x0        14      7168      3584         0  
//       x1        14       512       256         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 256, 512, 14, 14 }
// Out stride: { 0, 1, 7168, 512 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3584, 256 }
// Input 2 offset: 0
// Input 2 stride: { 512, 1, 0, 0 }
// Elementwise input X_I_61 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_60 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Sub)]] X_T243 = sub(X_T242, X_I_61)
// Elementwise op: [[pid(Mul)]] X_T244 = mul(X_T243, X_I_60)
// Tile size: { 64, 64, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Computed true ops: 102760448
// Computed work groups: 32
// Computed inner loops: 4
// Computed shared mem: 31216
// Computed out regs: 14336
// Computed mem read: 31616
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 8, 1
__kernel void kernel_c25_sdk_60(__global float* restrict  X_T244, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_61, __global const float* restrict  X_I_60)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3644];
  __local float in2_shared[4160];
  int co_gid = (get_group_id(1) * 64);
  int x1_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 256; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 256));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0_lid));
        int gidx = (((gbase + ci_tid) + (256 * x1_tid)) + (3584 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)50175)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 512));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (512 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)131071)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (911 * x1_tid)) + (65 * x0))];
          float val2 = in2_shared[(co + (65 * ci_lid))];
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
  int x1_cond = ((x1_gid != 12) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int co_lid = 0; co_lid < 2; co_lid += 1)
    {
      int co = ((32 * co_lid) + co_tid);
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T242 = agg[(co_lid + (x0_lid * 2))];
        int gout_idx = (((co_gid + co) + (7168 * x0)) + (512 * (x1_gid + x1_tid)));
        float LX_I_61 = X_I_61[(co_gid + co)];
        float LX_I_60 = X_I_60[(co_gid + co)];
        float LX_T243 = (LX_T242 - LX_I_61);
        float LX_T244 = (LX_T243 * LX_I_60);
        X_T244[gout_idx] = LX_T244;
      }
    }
  }
}
