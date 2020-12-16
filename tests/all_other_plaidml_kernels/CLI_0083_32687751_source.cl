#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 19 1
// lid: 256 1 1
// Original:
// X_T31[n, x0, x1, co : _T2, _T3, _T4, _T5] = +(X_I_31[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_32[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T31[n, x0, x1, co : _T2, _T3, _T4, _T5] = +(X_I_31[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_32[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= ci < 3, 0 <= co < 32, 0 <= co < 32, 0 <= x0 < 149, 0 <= x1 < 149, 0 <= k0 + 2*x0 < 299, 0 <= k1 + 2*x1 < 299, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= co < 32, 0 <= x0 < 149, 0 <= x1 < 149, 0 <= k0 + 2*x0 < 299, 0 <= k1 + 2*x1 < 299 }
// Defracted:
// X_T31[n, x0, x1, co : _T2, _T3, _T4, _T5] = +(X_I_31[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_32[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T31    X_I_31    X_I_32  
//       ci         3         0         1        32  
//       co        32         1         0         1  
//       k0         3         0       897       288  
//       k1         3         0         3        96  
//       x0       149      4768      1794         0  
//       x1       149        32         6         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, k0, k1, x0, x1 }
// Ranges: { 3, 32, 3, 3, 149, 149 }
// Out stride: { 0, 1, 0, 0, 4768, 32 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 897, 3, 1794, 6 }
// Input 2 offset: 0
// Input 2 stride: { 32, 1, 288, 96, 0, 0 }
// Elementwise input X_I_30 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Sub)]] X_T32 = sub(X_T31, X_I_30)
// Tile size: { 3, 32, 3, 3, 16, 8 }
// Contraction output var shape: fp32(1, 149, 149, 32):(710432, 4768, 32, 1):2775.12 KiB
// Computed true ops: 57544992
// Computed work groups: 190
// Computed inner loops: 1
// Computed shared mem: 10188
// Computed out regs: 16384
// Computed mem read: 10624
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 19, 1
__kernel void kernel_c51_sdk_0(__global float* restrict  X_T32, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_30)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1683];
  __local float in2_shared[864];
  int x1_gid = (get_group_id(1) * 8);
  int x0_gid = (get_group_id(0) * 16);
  for (int ci_gid = 0; ci_gid < 3; ci_gid += 3)
  {
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
    {
      for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
      {
        {
          int gbase = ((((ci_gid + (k1_gid * 3)) + (x1_gid * 6)) + (k0_gid * 897)) + (x0_gid * 1794));
          int ci_k1_x1_tid = (tid % 64);
          int k0_x0_tid = ((tid / 64) % 4);
          int ci_k1_x1_cond = (ci_k1_x1_tid < 51);
          if (ci_k1_x1_cond)
          {
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int k0_x0_cond = ((k0_x0_lid < 8) || (k0_x0_tid < 1));
              if (k0_x0_cond)
              {
                int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
                int lidx = (ci_k1_x1_tid + (51 * k0_x0));
                int gidx = ((gbase + ci_k1_x1_tid) + (897 * k0_x0));
                in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)268202)];
              }
            }
          }
        }
        {
          int gbase = (((ci_gid * 32) + (k1_gid * 96)) + (k0_gid * 288));
          int co_ci_k1_k0_tid = (tid % 256);
          for (int co_ci_k1_k0_lid = 0; co_ci_k1_k0_lid < 4; co_ci_k1_k0_lid += 1)
          {
            int co_ci_k1_k0_cond = ((co_ci_k1_k0_lid < 3) || (co_ci_k1_k0_tid < 96));
            if (co_ci_k1_k0_cond)
            {
              int co_ci_k1_k0 = ((256 * co_ci_k1_k0_lid) + co_ci_k1_k0_tid);
              int gidx = (gbase + co_ci_k1_k0);
              in2_shared[co_ci_k1_k0] = in2[clamp((int)gidx, (int)0, (int)863)];
            }
          }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int ci_lid = 0; ci_lid < 3; ci_lid += 1)
        {
          for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
          {
            for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
            {
              int co_tid = (tid % 32);
              int x1_tid = ((tid / 32) % 4);
              int x0_tid = ((tid / 128) % 2);
              for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((ci_lid + (3 * k1_lid)) + (6 * x1)) + (51 * k0_lid)) + (102 * x0))];
                  float val2 = in2_shared[(((co_tid + (32 * ci_lid)) + (96 * k1_lid)) + (288 * k0_lid))];
                  int agg_idx = (x1_lid + (x0_lid * 2));
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
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 1) || ((x1_gid != 144) || (x1_tid < 1)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 2) || ((x0_gid != 144) || (x0_tid < 1))) && ((x0_lid < 3) || (x0_gid != 144)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T31 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = ((co_tid + (4768 * (x0_gid + x0))) + (32 * (x1_gid + x1)));
          float LX_I_30 = X_I_30[co_tid];
          float LX_T32 = (LX_T31 - LX_I_30);
          X_T32[gout_idx] = LX_T32;
        }
      }
    }
  }
}
