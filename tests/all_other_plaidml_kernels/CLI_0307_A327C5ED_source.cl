#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Original:
// X_T575[n, x0, x1, c : _T753, _T754, _T755, _T756] = +(X_T574[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T575[n, x0, x1, c : _T753, _T754, _T755, _T756] = +(X_T574[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 256, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 256 }
// Defracted:
// X_T575[n, x0, x1, c : _T753, _T754, _T755, _T756] = +(X_T574[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T575    X_T574  
//        c       256         1         1  
//       k0         2         0      7168  
//       k1         2         0       256  
//       x0        14      3584     14336  
//       x1        14       256       512  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 256, 2, 2, 14, 14 }
// Out stride: { 1, 0, 0, 3584, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 256, 14336, 512 }
// Elementwise input X_T573 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_I_212 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_211 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(pool3_pool)]] X_T576 = div(X_T573, X_T575)
// Elementwise op: [[pid(Sub)]] X_T582 = sub(X_T576, X_I_212)
// Elementwise op: [[pid(Mul)]] X_T583 = mul(X_T582, X_I_211)
// Tile size: { 64, 2, 2, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 1003520
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 14568
// Computed out regs: 4096
// Computed mem read: 14672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 4, 1
__kernel void kernel_c124_sdk_179(__global float* restrict  X_T576, __global float* restrict  X_T583, __global const float* restrict  in1, __global const float* restrict  X_T573, __global const float* restrict  X_I_212, __global const float* restrict  X_I_211)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3642];
  int c_gid = (get_group_id(1) * 64);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 2)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 2)
    {
      {
        int gbase = (((c_gid + (k1_gid * 256)) + (x1_gid * 512)) + (k0_gid * 7168));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 2);
        int k0_x0_tid = ((tid / 128) % 2);
        for (int k0_x0_lid = 0; k0_x0_lid < 14; k0_x0_lid += 1)
        {
          int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
          int lidx = ((c_tid + (1821 * k1_x1_tid)) + (65 * k0_x0));
          int gidx = (((gbase + c_tid) + (256 * k1_x1_tid)) + (7168 * k0_x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 2; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 2; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x0_tid = ((tid / 32) % 8);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
            int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c = ((32 * c_lid) + c_tid);
              float val1 = in1_shared[(((c + (1821 * k1_lid)) + (65 * k0_lid)) + (130 * x0))];
              int agg_idx = (c_lid + (x0_lid * 2));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
    if (x0_cond)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T575 = agg[(c_lid + (x0_lid * 2))];
        int gout_idx = (((c_gid + c) + (3584 * x0)) + (256 * x1_gid));
        float LX_T573 = X_T573[gout_idx];
        float LX_I_212 = X_I_212[(c_gid + c)];
        float LX_I_211 = X_I_211[(c_gid + c)];
        float LX_T576 = (LX_T573 / LX_T575);
        float LX_T582 = (LX_T576 - LX_I_212);
        float LX_T583 = (LX_T582 * LX_I_211);
        X_T576[gout_idx] = LX_T576;
        X_T583[gout_idx] = LX_T583;
      }
    }
  }
}
