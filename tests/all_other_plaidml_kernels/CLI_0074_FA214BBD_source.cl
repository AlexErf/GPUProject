#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T75[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T74[n, k0 + x0, k1 + x1, ci] * X_I_69[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T75[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T74[n, k0 + x0, k1 + x1, ci] * X_I_69[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 42, 0 <= co < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + x0 < 83, 0 <= k1 + x1 < 83, 0 <= ci < 96, 0 <= ci < 96, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + x0 < 83, 0 <= k1 + x1 < 83, 0 <= ci < 96 }
// Defracted:
// X_T75[n, x0, x1, co : _T60, _T61, _T62, _T63] = +(X_T74[n, k0 + x0, k1 + x1, ci] * X_I_69[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T75     X_T74    X_I_69  
//       ci        96         0         1        42  
//       co        42         1         0         1  
//       x0        83      3486      7968         0  
//       x1        83        42        96         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 96, 42, 83, 83 }
// Out stride: { 0, 1, 3486, 42 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 7968, 96 }
// Input 2 offset: 0
// Input 2 stride: { 42, 1, 0, 0 }
// Elementwise input X_I_68 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_67 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(Sub)]] X_T76 = sub(X_T75, X_I_68)
// Elementwise op: [[pid(Mul)]] X_T77 = mul(X_T76, X_I_67)
// Tile size: { 32, 42, 8, 8 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 111105792
// Computed work groups: 121
// Computed inner loops: 3
// Computed shared mem: 13856
// Computed out regs: 16384
// Computed mem read: 14592
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_12(__global float* restrict  X_T77, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_68, __global const float* restrict  X_I_67)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2120];
  __local float in2_shared[1344];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int ci_gid = 0; ci_gid < 96; ci_gid += 32)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 96)) + (x0_gid * 7968));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 8);
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int lidx = ((ci_tid + (33 * x1_tid)) + (265 * x0_lid));
        int gidx = (((gbase + ci_tid) + (96 * x1_tid)) + (7968 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)661343)];
      }
    }
    {
      int gbase = (ci_gid * 42);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 6; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 5) || (co_ci_tid < 64));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)4031)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int co_lid = 0; co_lid < 2; co_lid += 1)
          {
            int co_cond = ((co_lid < 1) || (co_tid < 10));
            int co = select((int)0, (int)((32 * co_lid) + co_tid), (int)co_cond);
            float val1 = in1_shared[((ci_lid + (33 * x1)) + (265 * x0))];
            float val2 = in2_shared[(co + (42 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 4));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)co_cond);
          }
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
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 80) || (x1_tid < 3))) && ((x1_lid < 1) || (x1_gid != 80)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 1) || ((x0_gid != 80) || (x0_tid < 1))) && ((x0_lid < 2) || (x0_gid != 80)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int co_lid = 0; co_lid < 2; co_lid += 1)
          {
            int co_cond = ((co_lid < 1) || (co_tid < 10));
            if (co_cond)
            {
              int co = ((32 * co_lid) + co_tid);
              float LX_T75 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 4))];
              int gout_idx = ((co + (3486 * (x0_gid + x0))) + (42 * (x1_gid + x1)));
              float LX_I_68 = X_I_68[co];
              float LX_I_67 = X_I_67[co];
              float LX_T76 = (LX_T75 - LX_I_68);
              float LX_T77 = (LX_T76 * LX_I_67);
              X_T77[gout_idx] = LX_T77;
            }
          }
        }
      }
    }
  }
}
