#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T88[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T87[n, k0 + x0, k1 + x1, ci] * X_I_64[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T88[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T87[n, k0 + x0, k1 + x1, ci] * X_I_64[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 42, 0 <= ci < 42, 0 <= ci < 42, 0 <= co < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + x0 < 83, 0 <= k1 + x1 < 83, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 42, 0 <= ci < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + x0 < 83, 0 <= k1 + x1 < 83 }
// Defracted:
// X_T88[n, x0, x1, co : _T80, _T81, _T82, _T83] = +(X_T87[n, k0 + x0, k1 + x1, ci] * X_I_64[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T88     X_T87    X_I_64  
//       ci        42         0         1        42  
//       co        42         1         0         1  
//       x0        83      3486      3486         0  
//       x1        83        42        42         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 42, 42, 83, 83 }
// Out stride: { 0, 1, 3486, 42 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3486, 42 }
// Input 2 offset: 0
// Input 2 stride: { 42, 1, 0, 0 }
// Elementwise input X_I_63 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_62 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(Sub)]] X_T89 = sub(X_T88, X_I_63)
// Elementwise op: [[pid(Mul)]] X_T90 = mul(X_T89, X_I_62)
// Tile size: { 42, 42, 8, 8 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 48608784
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 17840
// Computed out regs: 16384
// Computed mem read: 18816
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_16(__global float* restrict  X_T90, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_63, __global const float* restrict  X_I_62)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2696];
  __local float in2_shared[1764];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int ci_gid = 0; ci_gid < 42; ci_gid += 42)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 42)) + (x0_gid * 3486));
      int ci_x1_tid = (tid % 256);
      for (int ci_x1_lid = 0; ci_x1_lid < 2; ci_x1_lid += 1)
      {
        int ci_x1_cond = ((ci_x1_lid < 1) || (ci_x1_tid < 80));
        if (ci_x1_cond)
        {
          int ci_x1 = ((256 * ci_x1_lid) + ci_x1_tid);
          for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
          {
            int lidx = (ci_x1 + (337 * x0_lid));
            int gidx = ((gbase + ci_x1) + (3486 * x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)289337)];
          }
        }
      }
    }
    {
      int gbase = (ci_gid * 42);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 7; co_ci_lid += 1)
      {
        int co_ci_cond = ((co_ci_lid < 6) || (co_ci_tid < 228));
        if (co_ci_cond)
        {
          int co_ci = ((256 * co_ci_lid) + co_ci_tid);
          int gidx = (gbase + co_ci);
          in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)1763)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 42; ci_lid += 1)
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
            float val1 = in1_shared[((ci_lid + (42 * x1)) + (337 * x0))];
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
              float LX_T88 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 4))];
              int gout_idx = ((co + (3486 * (x0_gid + x0))) + (42 * (x1_gid + x1)));
              float LX_I_63 = X_I_63[co];
              float LX_I_62 = X_I_62[co];
              float LX_T89 = (LX_T88 - LX_I_63);
              float LX_T90 = (LX_T89 * LX_I_62);
              X_T90[gout_idx] = LX_T90;
            }
          }
        }
      }
    }
  }
}
