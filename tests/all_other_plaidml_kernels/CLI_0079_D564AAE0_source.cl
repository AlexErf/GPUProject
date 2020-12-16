#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 28 1
// lid: 256 1 1
// Original:
// X_T93[n, x0, x1, co : _T73, _T74, _T75, _T76] = +(X_T92[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T93[n, x0, x1, co : _T73, _T74, _T75, _T76] = +(X_T92[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 16, 0 <= ci < 16, 0 <= co < 96, 0 <= co < 96, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 16, 0 <= co < 96, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112 }
// Defracted:
// X_T93[n, x0, x1, co : _T73, _T74, _T75, _T76] = +(X_T92[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T93     X_T92    X_I_77  
//       ci        16         0         1        96  
//       co        96         1         0         1  
//       x0       112     10752      1792         0  
//       x1       112        96        16         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 16, 96, 112, 112 }
// Out stride: { 0, 1, 10752, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 1792, 16 }
// Input 2 offset: 0
// Input 2 stride: { 96, 1, 0, 0 }
// Elementwise input X_I_76 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_75 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Sub)]] X_T94 = sub(X_T93, X_I_76)
// Elementwise op: [[pid(Mul)]] X_T95 = mul(X_T94, X_I_75)
// Tile size: { 16, 96, 4, 8 }
// Contraction output var shape: fp32(1, 112, 112, 96):(1204224, 10752, 96, 1):4704 KiB
// Computed true ops: 77070336
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 8208
// Computed out regs: 12288
// Computed mem read: 8960
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 28, 1
__kernel void kernel_c43_sdk_18(__global float* restrict  X_T95, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_76, __global const float* restrict  X_I_75)
{
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[516];
  __local float in2_shared[1536];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 16; ci_gid += 16)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 16)) + (x0_gid * 1792));
      int ci_x1_tid = (tid % 128);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        int lidx = (ci_x1_tid + (129 * x0));
        int gidx = ((gbase + ci_x1_tid) + (1792 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
      }
    }
    {
      int gbase = (ci_gid * 96);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 6; co_ci_lid += 1)
      {
        int co_ci = ((256 * co_ci_lid) + co_ci_tid);
        int gidx = (gbase + co_ci);
        in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)1535)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 3; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (16 * x1)) + (129 * x0))];
            float val2 = in2_shared[(co + (96 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 3)) + (x0_lid * 6));
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
  for (int co_lid = 0; co_lid < 3; co_lid += 1)
  {
    int co = ((32 * co_lid) + co_tid);
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T93 = agg[((co_lid + (x1_lid * 3)) + (x0_lid * 6))];
        int gout_idx = ((co + (10752 * (x0_gid + x0))) + (96 * (x1_gid + x1)));
        float LX_I_76 = X_I_76[co];
        float LX_I_75 = X_I_75[co];
        float LX_T94 = (LX_T93 - LX_I_76);
        float LX_T95 = (LX_T94 * LX_I_75);
        X_T95[gout_idx] = LX_T95;
      }
    }
  }
}
