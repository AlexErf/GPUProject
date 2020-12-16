#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 1 1
// lid: 256 1 1
// Original:
// X_T84[n, x0, x1, co : _T61, _T62, _T63, _T64] = +(X_T83[n, k0 + x0, k1 + x1, ci] * X_I_81[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T84[n, x0, x1, co : _T61, _T62, _T63, _T64] = +(X_T83[n, k0 + x0, k1 + x1, ci] * X_I_81[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 16, 0 <= co < 16, 0 <= ci < 32, 0 <= ci < 32, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 16, 0 <= ci < 32, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + x0 < 112, 0 <= k1 + x1 < 112 }
// Defracted:
// X_T84[n, x0, x1, co : _T61, _T62, _T63, _T64] = +(X_T83[n, k0 + x0, k1 + x1, ci] * X_I_81[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T84     X_T83    X_I_81  
//       ci        32         0         1        16  
//       co        16         1         0         1  
//       x0       112      1792      3584         0  
//       x1       112        16        32         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 32, 16, 112, 112 }
// Out stride: { 0, 1, 1792, 16 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3584, 32 }
// Input 2 offset: 0
// Input 2 stride: { 16, 1, 0, 0 }
// Elementwise input X_I_80 shape: fp32(16):(1):64 bytes
// Elementwise input X_I_79 shape: fp32(16):(1):64 bytes
// Elementwise op: [[pid(Sub)]] X_T85 = sub(X_T84, X_I_80)
// Elementwise op: [[pid(Mul)]] X_T86 = mul(X_T85, X_I_79)
// Tile size: { 32, 16, 112, 2 }
// Contraction output var shape: fp32(1, 112, 112, 16):(200704, 1792, 16, 1):784 KiB
// Computed true ops: 25690112
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 30976
// Computed out regs: 14336
// Computed mem read: 32512
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 1, 1
__kernel void kernel_c43_sdk_15(__global float* restrict  X_T86, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_80, __global const float* restrict  X_I_79)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7232];
  __local float in2_shared[512];
  int x1_gid = (get_group_id(0) * 2);
  for (int ci_gid = 0; ci_gid < 32; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 32));
      int ci_x1_tid = (tid % 64);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 28; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        int lidx = ((113 * ci_x1_tid) + x0);
        int gidx = ((gbase + ci_x1_tid) + (3584 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)401407)];
      }
    }
    {
      int gbase = (ci_gid * 16);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 2; co_ci_lid += 1)
      {
        int co_ci = ((256 * co_ci_lid) + co_ci_tid);
        int gidx = (gbase + co_ci);
        in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)511)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 16);
      int x1_tid = ((tid / 16) % 2);
      int x0_tid = ((tid / 32) % 8);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0 = ((8 * x0_lid) + x0_tid);
        float val1 = in1_shared[(((113 * ci_lid) + (3616 * x1_tid)) + x0)];
        float val2 = in2_shared[(co_tid + (16 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = agg_rhs;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 2);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
  {
    int x0 = ((8 * x0_lid) + x0_tid);
    float LX_T84 = agg[x0_lid];
    int gout_idx = ((co_tid + (1792 * x0)) + (16 * (x1_gid + x1_tid)));
    float LX_I_80 = X_I_80[co_tid];
    float LX_I_79 = X_I_79[co_tid];
    float LX_T85 = (LX_T84 - LX_I_80);
    float LX_T86 = (LX_T85 * LX_I_79);
    X_T86[gout_idx] = LX_T86;
  }
}
