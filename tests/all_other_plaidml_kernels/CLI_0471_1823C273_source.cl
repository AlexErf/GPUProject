#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 4 1
// lid: 256 1 1
// Original:
// X_T1383[n, x0, x1, co : _T1954, _T1955, _T1956, _T1957] = +(X_T1382[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T1383[n, x0, x1, co : _T1954, _T1955, _T1956, _T1957] = +(X_T1382[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 640, 0 <= co < 640, 0 <= ci < 1280, 0 <= ci < 1280, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 640, 0 <= ci < 1280 }
// Defracted:
// X_T1383[n, x0, x1, co : _T1954, _T1955, _T1956, _T1957] = +(X_T1382[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1383   X_T1382     X_I_5  
//       ci      1280         0         1       640  
//       co       640         1         0         1  
//       x0        14      8960     17920         0  
//       x1        14       640      1280         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 1280, 640, 14, 14 }
// Out stride: { 0, 1, 8960, 640 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 17920, 1280 }
// Input 2 offset: 0
// Input 2 stride: { 640, 1, 0, 0 }
// Tile size: { 64, 64, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Computed true ops: 321126400
// Computed work groups: 40
// Computed inner loops: 20
// Computed shared mem: 31216
// Computed out regs: 14336
// Computed mem read: 30720
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 4, 1
__kernel void kernel_c108_sdk_470(__global float* restrict  X_T1383, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3644];
  __local float in2_shared[4160];
  int co_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 1280; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 1280));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0_lid));
        int gidx = (((gbase + ci_tid) + (1280 * x1_tid)) + (17920 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)250879)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 640));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (640 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)819199)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (911 * x1_tid)) + (65 * x0))];
          float val2 = in2_shared[(co + (65 * ci_lid))];
          int agg_idx = (co_lid + (x0_lid * 2));
          float agg_rhs = mad(val2, val1, agg[agg_idx]);
          agg[agg_idx] = agg_rhs;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int x1_cond = ((x1_gid != 12) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int co_lid = 0; co_lid < 2; co_lid += 1)
    {
      int co = ((32 * co_lid) + co_tid);
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T1383 = agg[(co_lid + (x0_lid * 2))];
        int gout_idx = (((co_gid + co) + (8960 * x0)) + (640 * (x1_gid + x1_tid)));
        X_T1383[gout_idx] = LX_T1383;
      }
    }
  }
}
