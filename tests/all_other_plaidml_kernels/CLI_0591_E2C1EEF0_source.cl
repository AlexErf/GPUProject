#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Original:
// X_T1791[n, x0, x1, co : _T2546, _T2547, _T2548, _T2549] = +(X_T1790[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T1791[n, x0, x1, co : _T2546, _T2547, _T2548, _T2549] = +(X_T1790[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 896, 0 <= co < 896, 0 <= ci < 1792, 0 <= ci < 1792, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 896, 0 <= ci < 1792 }
// Defracted:
// X_T1791[n, x0, x1, co : _T2546, _T2547, _T2548, _T2549] = +(X_T1790[n, k0 + x0, k1 + x1, ci] * X_I_5[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1791   X_T1790     X_I_5  
//       ci      1792         0         1       896  
//       co       896         1         0         1  
//       x0        14     12544     25088         0  
//       x1        14       896      1792         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 1792, 896, 14, 14 }
// Out stride: { 0, 1, 12544, 896 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 25088, 1792 }
// Input 2 offset: 0
// Input 2 stride: { 896, 1, 0, 0 }
// Tile size: { 64, 64, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Computed true ops: 629407744
// Computed work groups: 56
// Computed inner loops: 28
// Computed shared mem: 31216
// Computed out regs: 14336
// Computed mem read: 30720
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 4, 1
__kernel void kernel_c124_sdk_614(__global float* restrict  X_T1791, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3644];
  __local float in2_shared[4160];
  int co_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 1792; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 1792));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0_lid));
        int gidx = (((gbase + ci_tid) + (1792 * x1_tid)) + (25088 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)351231)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 896));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (896 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)1605631)];
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
        float LX_T1791 = agg[(co_lid + (x0_lid * 2))];
        int gout_idx = (((co_gid + co) + (12544 * x0)) + (896 * (x1_gid + x1_tid)));
        X_T1791[gout_idx] = LX_T1791;
      }
    }
  }
}
