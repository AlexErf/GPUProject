#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 2 1
// lid: 256 1 1
// Original:
// X_T2002[n, x0, x1, co : _T2809, _T2810, _T2811, _T2812] = +(X_T2000[n, k0 + x0, k1 + x1, ci] * X_I_722[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T2002[n, x0, x1, co : _T2809, _T2810, _T2811, _T2812] = +(X_T2000[n, k0 + x0, k1 + x1, ci] * X_I_722[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + x0 < 8, 0 <= k1 + x1 < 8, 0 <= co < 192, 0 <= co < 192, 0 <= ci < 2080, 0 <= ci < 2080, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + x0 < 8, 0 <= k1 + x1 < 8, 0 <= co < 192, 0 <= ci < 2080 }
// Defracted:
// X_T2002[n, x0, x1, co : _T2809, _T2810, _T2811, _T2812] = +(X_T2000[n, k0 + x0, k1 + x1, ci] * X_I_722[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2002   X_T2000   X_I_722  
//       ci      2080         0         1       192  
//       co       192         1         0         1  
//       x0         8      1536     16640         0  
//       x1         8       192      2080         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 2080, 192, 8, 8 }
// Out stride: { 0, 1, 1536, 192 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 16640, 2080 }
// Input 2 offset: 0
// Input 2 stride: { 192, 1, 0, 0 }
// Elementwise input X_I_721 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Sub)]] X_T2003 = sub(X_T2002, X_I_721)
// Tile size: { 32, 32, 8, 4 }
// Contraction output var shape: fp32(1, 8, 8, 192):(12288, 1536, 192, 1):48 KiB
// Computed true ops: 76677120
// Computed work groups: 12
// Computed inner loops: 65
// Computed shared mem: 8464
// Computed out regs: 4096
// Computed mem read: 8320
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 2, 1
__kernel void kernel_c51_sdk_655(__global float* restrict  X_T2003, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_721)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1060];
  __local float in2_shared[1056];
  int co_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 2080; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x1_gid * 2080));
      int ci_tid = (tid % 32);
      int x0_tid = ((tid / 32) % 8);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int lidx = ((ci_tid + (33 * x0_tid)) + (265 * x1_lid));
        int gidx = (((gbase + ci_tid) + (16640 * x0_tid)) + (2080 * x1_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)133119)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 192));
      int co_tid = (tid % 32);
      int ci_tid = ((tid / 32) % 8);
      for (int ci_lid = 0; ci_lid < 4; ci_lid += 1)
      {
        int ci = ((8 * ci_lid) + ci_tid);
        int lidx = (co_tid + (33 * ci));
        int gidx = ((gbase + co_tid) + (192 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)399359)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float val1 = in1_shared[((ci_lid + (265 * x1_tid)) + (33 * x0))];
        float val2 = in2_shared[(co_tid + (33 * ci_lid))];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = agg_rhs;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0 = ((2 * x0_lid) + x0_tid);
    float LX_T2002 = agg[x0_lid];
    int gout_idx = (((co_gid + co_tid) + (1536 * x0)) + (192 * (x1_gid + x1_tid)));
    float LX_I_721 = X_I_721[(co_gid + co_tid)];
    float LX_T2003 = (LX_T2002 - LX_I_721);
    X_T2003[gout_idx] = LX_T2003;
  }
}
