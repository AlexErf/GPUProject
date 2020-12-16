#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 21
// lid: 256 1 1
// Original:
// X_T37[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_57[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_58[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T37[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_57[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_58[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= ci < 3, 0 <= co < 96, 0 <= co < 96, 0 <= x0 < 165, 0 <= x1 < 165, 0 <= k0 + 2*x0 < 331, 0 <= k1 + 2*x1 < 331, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= ci < 3, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= co < 96, 0 <= x0 < 165, 0 <= x1 < 165, 0 <= k0 + 2*x0 < 331, 0 <= k1 + 2*x1 < 331 }
// Defracted:
// X_T37[n, x0, x1, co : _T9, _T10, _T11, _T12] = +(X_I_57[n, k0 + 2*x0, k1 + 2*x1, ci] * X_I_58[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T37    X_I_57    X_I_58  
//       ci         3         0         1        96  
//       co        96         1         0         1  
//       k0         3         0       993       864  
//       k1         3         0         3       288  
//       x0       165     15840      1986         0  
//       x1       165        96         6         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, k0, k1, x0, x1 }
// Ranges: { 3, 96, 3, 3, 165, 165 }
// Out stride: { 0, 1, 0, 0, 15840, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 993, 3, 1986, 6 }
// Input 2 offset: 0
// Input 2 stride: { 96, 1, 864, 288, 0, 0 }
// Elementwise input X_I_56 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_55 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Sub)]] X_T38 = sub(X_T37, X_I_56)
// Elementwise op: [[pid(Mul)]] X_T39 = mul(X_T38, X_I_55)
// Tile size: { 3, 32, 3, 3, 16, 8 }
// Contraction output var shape: fp32(1, 165, 165, 96):(2613600, 15840, 96, 1):10209.4 KiB
// Computed true ops: 282268800
// Computed work groups: 693
// Computed inner loops: 1
// Computed shared mem: 10188
// Computed out regs: 16384
// Computed mem read: 11136
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 21
__kernel void kernel_c42_sdk_0(__global float* restrict  X_T39, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_56, __global const float* restrict  X_I_55)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1683];
  __local float in2_shared[864];
  int co_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(2) * 8);
  int x0_gid = (get_group_id(1) * 16);
  for (int ci_gid = 0; ci_gid < 3; ci_gid += 3)
  {
    for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
    {
      for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
      {
        {
          int gbase = ((((ci_gid + (k1_gid * 3)) + (x1_gid * 6)) + (k0_gid * 993)) + (x0_gid * 1986));
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
                int gidx = ((gbase + ci_k1_x1_tid) + (993 * k0_x0));
                in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)328682)];
              }
            }
          }
        }
        {
          int gbase = (((co_gid + (ci_gid * 96)) + (k1_gid * 288)) + (k0_gid * 864));
          int co_tid = (tid % 32);
          int ci_k1_k0_tid = ((tid / 32) % 8);
          for (int ci_k1_k0_lid = 0; ci_k1_k0_lid < 4; ci_k1_k0_lid += 1)
          {
            int ci_k1_k0_cond = ((ci_k1_k0_lid < 3) || (ci_k1_k0_tid < 3));
            if (ci_k1_k0_cond)
            {
              int ci_k1_k0 = ((8 * ci_k1_k0_lid) + ci_k1_k0_tid);
              int lidx = ((27 * co_tid) + ci_k1_k0);
              int gidx = ((gbase + co_tid) + (96 * ci_k1_k0));
              in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)2591)];
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
                  float val2 = in2_shared[((((27 * co_tid) + ci_lid) + (3 * k1_lid)) + (9 * k0_lid))];
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
    int x1_cond = ((x1_lid < 1) || ((x1_gid != 160) || (x1_tid < 1)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 2) || ((x0_gid != 160) || (x0_tid < 1))) && ((x0_lid < 3) || (x0_gid != 160)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T37 = agg[(x1_lid + (x0_lid * 2))];
          int gout_idx = (((co_gid + co_tid) + (15840 * (x0_gid + x0))) + (96 * (x1_gid + x1)));
          float LX_I_56 = X_I_56[(co_gid + co_tid)];
          float LX_I_55 = X_I_55[(co_gid + co_tid)];
          float LX_T38 = (LX_T37 - LX_I_56);
          float LX_T39 = (LX_T38 * LX_I_55);
          X_T39[gout_idx] = LX_T39;
        }
      }
    }
  }
}
