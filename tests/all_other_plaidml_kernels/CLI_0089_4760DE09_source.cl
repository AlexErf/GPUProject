#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 7
// lid: 256 1 1
// Original:
// X_T136[n, x0, x1, co : _T124, _T125, _T126, _T127] = +(X_T130[n, k0 + x0, k1 + x1, ci] * X_I_108[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T136[n, x0, x1, co : _T124, _T125, _T126, _T127] = +(X_T130[n, k0 + x0, k1 + x1, ci] * X_I_108[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 24, 0 <= ci < 24, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= co < 144, 0 <= co < 144, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 24, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= co < 144 }
// Defracted:
// X_T136[n, x0, x1, co : _T124, _T125, _T126, _T127] = +(X_T130[n, k0 + x0, k1 + x1, ci] * X_I_108[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T136    X_T130   X_I_108  
//       ci        24         0         1       144  
//       co       144         1         0         1  
//       x0        56      8064      1344         0  
//       x1        56       144        24         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 24, 144, 56, 56 }
// Out stride: { 0, 1, 8064, 144 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 1344, 24 }
// Input 2 offset: 0
// Input 2 stride: { 144, 1, 0, 0 }
// Elementwise input X_I_107 shape: fp32(144):(1):576 bytes
// Elementwise input X_I_106 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(Sub)]] X_T137 = sub(X_T136, X_I_107)
// Elementwise op: [[pid(Mul)]] X_T138 = mul(X_T137, X_I_106)
// Tile size: { 24, 64, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 144):(451584, 8064, 144, 1):1764 KiB
// Computed true ops: 43352064
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 12416
// Computed out regs: 16384
// Computed mem read: 13312
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 7
__kernel void kernel_c43_sdk_30(__global float* restrict  X_T138, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_107, __global const float* restrict  X_I_106)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1544];
  __local float in2_shared[1560];
  int co_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 8);
  int x0_gid = (get_group_id(2) * 8);
  for (int ci_gid = 0; ci_gid < 24; ci_gid += 24)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 24)) + (x0_gid * 1344));
      int ci_x1_tid = (tid % 256);
      int ci_x1_cond = (ci_x1_tid < 192);
      if (ci_x1_cond)
      {
        for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
        {
          int lidx = (ci_x1_tid + (193 * x0_lid));
          int gidx = ((gbase + ci_x1_tid) + (1344 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)75263)];
        }
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 144));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 6; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (144 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)3455)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 24; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (24 * x1)) + (193 * x0))];
            float val2 = in2_shared[(co + (65 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 4));
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
  for (int co_lid = 0; co_lid < 2; co_lid += 1)
  {
    int co_cond = (((co_lid < 0) || ((co_gid != 128) || (co_tid < 16))) && ((co_lid < 1) || (co_gid != 128)));
    if (co_cond)
    {
      int co = ((32 * co_lid) + co_tid);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T136 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 4))];
          int gout_idx = (((co_gid + co) + (8064 * (x0_gid + x0))) + (144 * (x1_gid + x1)));
          float LX_I_107 = X_I_107[(co_gid + co)];
          float LX_I_106 = X_I_106[(co_gid + co)];
          float LX_T137 = (LX_T136 - LX_I_107);
          float LX_T138 = (LX_T137 * LX_I_106);
          X_T138[gout_idx] = LX_T138;
        }
      }
    }
  }
}
