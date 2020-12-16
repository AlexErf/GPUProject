#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 2 4
// lid: 256 1 1
// Original:
// X_T33[n, x0, x1, co : _T6, _T7, _T8, _T9] = +(X_T32[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_246[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T33[n, x0, x1, co : _T6, _T7, _T8, _T9] = +(X_T32[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_246[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= ci < 3, 0 <= ci < 3, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= co < 64, 0 <= co < 64, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + 2*x0 < 230, 0 <= k1 + 2*x1 < 230, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= co < 64, 0 <= x0 < 112, 0 <= x1 < 112, 0 <= k0 + 2*x0 < 230, 0 <= k1 + 2*x1 < 230 }
// Defracted:
// X_T33[n, x0, x1, co : _T6, _T7, _T8, _T9] = +(X_T32[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_246[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T33     X_T32   X_I_246  
//       ci         3         0         1        64  
//       co        64         1         0         1  
//       k0         7         0       690      1344  
//       k1         7         0         3       192  
//       x0       112      7168      1380         0  
//       x1       112        64         6         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, k0, k1, x0, x1 }
// Ranges: { 3, 64, 7, 7, 112, 112 }
// Out stride: { 0, 1, 0, 0, 7168, 64 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 690, 3, 1380, 6 }
// Input 2 offset: 0
// Input 2 stride: { 64, 1, 1344, 192, 0, 0 }
// Elementwise input X_I_245 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_244 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_243 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T34 = add(X_T33, X_I_245)
// Elementwise op: [[pid(Sub)]] X_T35 = sub(X_T34, X_I_244)
// Elementwise op: [[pid(Mul)]] X_T36 = mul(X_T35, X_I_243)
// Tile size: { 3, 32, 7, 7, 4, 32 }
// Contraction output var shape: fp32(1, 112, 112, 64):(802816, 7168, 64, 1):3136 KiB
// Computed true ops: 590069760
// Computed work groups: 224
// Computed inner loops: 1
// Computed shared mem: 29580
// Computed out regs: 16384
// Computed mem read: 31104
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 2, 4
__kernel void kernel_c29_sdk_1(__global float* restrict  X_T36, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_245, __global const float* restrict  X_I_244, __global const float* restrict  X_I_243)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2691];
  __local float in2_shared[4704];
  int co_gid = (get_group_id(1) * 32);
  int x1_gid = (get_group_id(2) * 32);
  int x0_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 3; ci_gid += 3)
  {
    for (int k0_gid = 0; k0_gid < 7; k0_gid += 7)
    {
      for (int k1_gid = 0; k1_gid < 7; k1_gid += 7)
      {
        {
          int gbase = ((((ci_gid + (k1_gid * 3)) + (x1_gid * 6)) + (k0_gid * 690)) + (x0_gid * 1380));
          int ci_k1_x1_tid = (tid % 256);
          int ci_k1_x1_cond = (ci_k1_x1_tid < 207);
          if (ci_k1_x1_cond)
          {
            for (int k0_x0_lid = 0; k0_x0_lid < 13; k0_x0_lid += 1)
            {
              int lidx = (ci_k1_x1_tid + (207 * k0_x0_lid));
              int gidx = ((gbase + ci_k1_x1_tid) + (690 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)158699)];
            }
          }
        }
        {
          int gbase = (((co_gid + (ci_gid * 64)) + (k1_gid * 192)) + (k0_gid * 1344));
          int co_tid = (tid % 32);
          int ci_k1_k0_tid = ((tid / 32) % 8);
          for (int ci_k1_k0_lid = 0; ci_k1_k0_lid < 19; ci_k1_k0_lid += 1)
          {
            int ci_k1_k0_cond = ((ci_k1_k0_lid < 18) || (ci_k1_k0_tid < 3));
            if (ci_k1_k0_cond)
            {
              int ci_k1_k0 = ((8 * ci_k1_k0_lid) + ci_k1_k0_tid);
              int lidx = ((147 * co_tid) + ci_k1_k0);
              int gidx = ((gbase + co_tid) + (64 * ci_k1_k0));
              in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)9407)];
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ci_lid = 0; ci_lid < 3; ci_lid += 1)
        {
          for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
          {
            for (int k1_lid = 0; k1_lid < 7; k1_lid += 1)
            {
              int co_tid = (tid % 32);
              int x1_tid = ((tid / 32) % 4);
              int x0_tid = ((tid / 128) % 2);
              for (int x1_lid = 0; x1_lid < 8; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((ci_lid + (3 * k1_lid)) + (6 * x1)) + (207 * k0_lid)) + (414 * x0))];
                  float val2 = in2_shared[((((147 * co_tid) + ci_lid) + (3 * k1_lid)) + (21 * k0_lid))];
                  int agg_idx = (x1_lid + (x0_lid * 8));
                  float agg_rhs = mad(val2, val1, agg[agg_idx]);
                  agg[agg_idx] = agg_rhs;
                }
              }
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x1_lid = 0; x1_lid < 8; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 4) || (x1_gid != 96));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T33 = agg[(x1_lid + (x0_lid * 8))];
        int gout_idx = (((co_gid + co_tid) + (7168 * (x0_gid + x0))) + (64 * (x1_gid + x1)));
        float LX_I_245 = X_I_245[(co_gid + co_tid)];
        float LX_I_244 = X_I_244[(co_gid + co_tid)];
        float LX_I_243 = X_I_243[(co_gid + co_tid)];
        float LX_T34 = (LX_T33 + LX_I_245);
        float LX_T35 = (LX_T34 - LX_I_244);
        float LX_T36 = (LX_T35 * LX_I_243);
        X_T36[gout_idx] = LX_T36;
      }
    }
  }
}
