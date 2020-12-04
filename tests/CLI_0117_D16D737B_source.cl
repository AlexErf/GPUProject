#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T292[n, x0, x1, co : _T430, _T431, _T432, _T433] = +(X_T291[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T292[n, x0, x1, co : _T430, _T431, _T432, _T433] = +(X_T291[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= ci < 22, 0 <= ci < 22, 0 <= co < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 22, 0 <= ci < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28 }
// Defracted:
// X_T292[n, x0, x1, co : _T430, _T431, _T432, _T433] = +(X_T291[n, k0 + x0, k1 + x1, ci] * X_I_118[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T292    X_T291   X_I_118  
//       ci        22         0         1        22  
//       co        22         1         0         1  
//       x0        28       616       616         0  
//       x1        28        22        22         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 22, 22, 28, 28 }
// Out stride: { 0, 1, 616, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 616, 22 }
// Input 2 offset: 0
// Input 2 stride: { 22, 1, 0, 0 }
// Elementwise input X_I_117 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_116 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(Sub)]] X_T293 = sub(X_T292, X_I_117)
// Elementwise op: [[pid(Mul)]] X_T294 = mul(X_T293, X_I_116)
// Tile size: { 22, 22, 4, 8 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 1517824
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 4768
// Computed out regs: 4096
// Computed mem read: 4992
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c42_sdk_98(__global float* restrict  X_T294, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_117, __global const float* restrict  X_I_116)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[708];
  __local float in2_shared[484];
  int x1_gid = (get_group_id(1) * 8);
  int x0_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 22; ci_gid += 22)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 22)) + (x0_gid * 616));
      int ci_x1_tid = (tid % 256);
      int ci_x1_cond = (ci_x1_tid < 176);
      if (ci_x1_cond)
      {
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int lidx = (ci_x1_tid + (177 * x0_lid));
          int gidx = ((gbase + ci_x1_tid) + (616 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)17247)];
        }
      }
    }
    {
      int gbase = (ci_gid * 22);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 2; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 1) || (co_ci_tid < 228));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)483)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 22; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        int co_cond = (co_tid < 22);
        int co = select((int)0, (int)co_tid, (int)co_cond);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (22 * x1)) + (177 * x0))];
          float val2 = in2_shared[(co + (22 * ci_lid))];
          int agg_idx = (x1_lid + (x0_lid * 2));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)co_cond);
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 1) || (x1_gid != 24));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      int co_cond = (co_tid < 22);
      if (co_cond)
      {
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T292 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = ((co_tid + (616 * (x0_gid + x0))) + (22 * (x1_gid + x1)));
          float LX_I_117 = X_I_117[co_tid];
          float LX_I_116 = X_I_116[co_tid];
          float LX_T293 = (LX_T292 - LX_I_117);
          float LX_T294 = (LX_T293 * LX_I_116);
          X_T294[gout_idx] = LX_T294;
        }
      }
    }
  }
}
