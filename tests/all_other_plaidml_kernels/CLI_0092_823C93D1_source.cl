#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 19 1
// lid: 256 1 1
// Original:
// X_T63[n, x0, x1, co : _T48, _T49, _T50, _T51] = +(X_T62[n, k0 + x0, k1 + x1, ci] * X_I_22[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T63[n, x0, x1, co : _T48, _T49, _T50, _T51] = +(X_T62[n, k0 + x0, k1 + x1, ci] * X_I_22[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 64, 0 <= ci < 64, 0 <= x0 < 73, 0 <= x1 < 73, 0 <= k0 + x0 < 73, 0 <= k1 + x1 < 73, 0 <= co < 80, 0 <= co < 80, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= ci < 64, 0 <= x0 < 73, 0 <= x1 < 73, 0 <= k0 + x0 < 73, 0 <= k1 + x1 < 73, 0 <= co < 80 }
// Defracted:
// X_T63[n, x0, x1, co : _T48, _T49, _T50, _T51] = +(X_T62[n, k0 + x0, k1 + x1, ci] * X_I_22[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T63     X_T62    X_I_22  
//       ci        64         0         1        80  
//       co        80         1         0         1  
//       x0        73      5840      4672         0  
//       x1        73        80        64         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 64, 80, 73, 73 }
// Out stride: { 0, 1, 5840, 80 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 4672, 64 }
// Input 2 offset: 0
// Input 2 stride: { 80, 1, 0, 0 }
// Elementwise input X_I_21 shape: fp32(80):(1):320 bytes
// Elementwise op: [[pid(Sub)]] X_T64 = sub(X_T63, X_I_21)
// Tile size: { 64, 80, 4, 8 }
// Contraction output var shape: fp32(1, 73, 73, 80):(426320, 5840, 80, 1):1665.31 KiB
// Computed true ops: 81853440
// Computed work groups: 190
// Computed inner loops: 1
// Computed shared mem: 28688
// Computed out regs: 12288
// Computed mem read: 29056
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 19, 1
__kernel void kernel_c51_sdk_10(__global float* restrict  X_T64, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_21)
{
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2052];
  __local float in2_shared[5120];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 64; ci_gid += 64)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 64)) + (x0_gid * 4672));
      int ci_x1_tid = (tid % 256);
      for (int ci_x1_lid = 0; ci_x1_lid < 2; ci_x1_lid += 1)
      {
        int ci_x1 = ((256 * ci_x1_lid) + ci_x1_tid);
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int lidx = (ci_x1 + (513 * x0_lid));
          int gidx = ((gbase + ci_x1) + (4672 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)341055)];
        }
      }
    }
    {
      int gbase = (ci_gid * 80);
      int co_ci_tid = (tid % 256);
      for (int co_ci_lid = 0; co_ci_lid < 20; co_ci_lid += 1)
      {
        int co_ci = ((256 * co_ci_lid) + co_ci_tid);
        int gidx = (gbase + co_ci);
        in2_shared[co_ci] = in2[clamp((int)gidx, (int)0, (int)5119)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int co_lid = 0; co_lid < 3; co_lid += 1)
          {
            int co_cond = ((co_lid < 2) || (co_tid < 16));
            int co = select((int)0, (int)((32 * co_lid) + co_tid), (int)co_cond);
            float val1 = in1_shared[((ci_lid + (64 * x1)) + (513 * x0))];
            float val2 = in2_shared[(co + (80 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 3)) + (x0_lid * 6));
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
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 72) || (x1_tid < 1))) && ((x1_lid < 1) || (x1_gid != 72)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 0) || ((x0_gid != 72) || (x0_tid < 1))) && ((x0_lid < 1) || (x0_gid != 72)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int co_lid = 0; co_lid < 3; co_lid += 1)
          {
            int co_cond = ((co_lid < 2) || (co_tid < 16));
            if (co_cond)
            {
              int co = ((32 * co_lid) + co_tid);
              float LX_T63 = agg[((co_lid + (x1_lid * 3)) + (x0_lid * 6))];
              int gout_idx = ((co + (5840 * (x0_gid + x0))) + (80 * (x1_gid + x1)));
              float LX_I_21 = X_I_21[co];
              float LX_T64 = (LX_T63 - LX_I_21);
              X_T64[gout_idx] = LX_T64;
            }
          }
        }
      }
    }
  }
}
