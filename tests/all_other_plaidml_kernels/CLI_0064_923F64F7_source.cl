#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 14 1
// lid: 256 1 1
// Original:
// X_T34[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_45[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_46[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T34[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_45[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_46[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= ci < 3, 0 <= co < 32, 0 <= co < 32, 0 <= x0 < 111, 0 <= x1 < 111, 0 <= k0 + 2*x0 < 224, 0 <= k1 + 2*x1 < 224, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= co < 32, 0 <= x0 < 111, 0 <= x1 < 111, 0 <= k0 + 2*x0 < 224, 0 <= k1 + 2*x1 < 224 }
// Defracted:
// X_T34[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_45[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_46[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T34    X_I_45    X_I_46  
//       ci         3         0         1        32  
//       co        32         1         0         1  
//       k0         3         0       672       288  
//       k1         3         0         3        96  
//       x0       111      3552      1344         0  
//       x1       111        32         6         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, k0, k1, x0, x1 }
// Ranges: { 3, 32, 3, 3, 111, 111 }
// Out stride: { 0, 1, 0, 0, 3552, 32 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 672, 3, 1344, 6 }
// Input 2 offset: 0
// Input 2 stride: { 32, 1, 288, 96, 0, 0 }
// Elementwise input X_I_44 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_43 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Sub)]] X_T35 = sub(X_T34, X_I_44)
// Elementwise op: [[pid(Mul)]] X_T36 = mul(X_T35, X_I_43)
// Tile size: { 3, 32, 3, 3, 8, 16 }
// Contraction output var shape: fp32(1, 111, 111, 32):(394272, 3552, 32, 1):1540.12 KiB
// Computed true ops: 42581376
// Computed work groups: 98
// Computed inner loops: 1
// Computed shared mem: 10188
// Computed out regs: 16384
// Computed mem read: 11136
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 14, 1
__kernel void kernel_c42_sdk_0(__global float* restrict  X_T36, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_44, __global const float* restrict  X_I_43)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1683];
  __local float in2_shared[864];
  int x1_gid = (get_group_id(0) * 16);
  int x0_gid = (get_group_id(1) * 8);
  for (int ci_gid = 0; ci_gid < 3; ci_gid += 3)
  {
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
    {
      for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
      {
        {
          int gbase = ((((ci_gid + (k1_gid * 3)) + (x1_gid * 6)) + (k0_gid * 672)) + (x0_gid * 1344));
          int ci_k1_x1_tid = (tid % 128);
          int k0_x0_tid = ((tid / 128) % 2);
          int ci_k1_x1_cond = (ci_k1_x1_tid < 99);
          if (ci_k1_x1_cond)
          {
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int k0_x0_cond = ((k0_x0_lid < 8) || (k0_x0_tid < 1));
              if (k0_x0_cond)
              {
                int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
                int lidx = (ci_k1_x1_tid + (99 * k0_x0));
                int gidx = ((gbase + ci_k1_x1_tid) + (672 * k0_x0));
                in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)150527)];
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
              for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
              {
                int x1 = ((4 * x1_lid) + x1_tid);
                for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
                {
                  int x0 = ((2 * x0_lid) + x0_tid);
                  float val1 = in1_shared[((((ci_lid + (3 * k1_lid)) + (6 * x1)) + (99 * k0_lid)) + (198 * x0))];
                  float val2 = in2_shared[(((co_tid + (32 * ci_lid)) + (96 * k1_lid)) + (288 * k0_lid))];
                  int agg_idx = (x1_lid + (x0_lid * 4));
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
  for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 3) || ((x1_gid != 96) || (x1_tid < 3)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 3) || ((x0_gid != 104) || (x0_tid < 1)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T34 = agg[(x1_lid + (x0_lid * 4))];
          int gout_idx = ((co_tid + (3552 * (x0_gid + x0))) + (32 * (x1_gid + x1)));
          float LX_I_44 = X_I_44[co_tid];
          float LX_I_43 = X_I_43[co_tid];
          float LX_T35 = (LX_T34 - LX_I_44);
          float LX_T36 = (LX_T35 * LX_I_43);
          X_T36[gout_idx] = LX_T36;
        }
      }
    }
  }
}
