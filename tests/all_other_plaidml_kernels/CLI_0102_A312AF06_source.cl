#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Original:
// X_T216[n, x0, x1, co : _T222, _T223, _T224, _T225] = +(X_T210[n, k0 + x0, k1 + x1, ci] * X_I_126[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T216[n, x0, x1, co : _T222, _T223, _T224, _T225] = +(X_T210[n, k0 + x0, k1 + x1, ci] * X_I_126[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= ci < 32, 0 <= ci < 32, 0 <= co < 192, 0 <= co < 192, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= ci < 32, 0 <= co < 192 }
// Defracted:
// X_T216[n, x0, x1, co : _T222, _T223, _T224, _T225] = +(X_T210[n, k0 + x0, k1 + x1, ci] * X_I_126[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T216    X_T210   X_I_126  
//       ci        32         0         1       192  
//       co       192         1         0         1  
//       x0        28      5376       896         0  
//       x1        28       192        32         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 32, 192, 28, 28 }
// Out stride: { 0, 1, 5376, 192 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 896, 32 }
// Input 2 offset: 0
// Input 2 stride: { 192, 1, 0, 0 }
// Elementwise input X_I_125 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_124 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Sub)]] X_T217 = sub(X_T216, X_I_125)
// Elementwise op: [[pid(Mul)]] X_T218 = mul(X_T217, X_I_124)
// Tile size: { 32, 32, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Computed true ops: 19267584
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 18560
// Computed out regs: 14336
// Computed mem read: 19328
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c43_sdk_52(__global float* restrict  X_T218, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_125, __global const float* restrict  X_I_124)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3584];
  __local float in2_shared[1056];
  int co_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 32; ci_gid += 32)
  {
    {
      int gbase = (ci_gid + (x0_gid * 896));
      int ci_x1_x0_tid = (tid % 256);
      for (int ci_x1_x0_lid = 0; ci_x1_x0_lid < 14; ci_x1_x0_lid += 1)
      {
        int ci_x1_x0 = ((256 * ci_x1_x0_lid) + ci_x1_x0_tid);
        int gidx = (gbase + ci_x1_x0);
        in1_shared[ci_x1_x0] = in1[clamp((int)gidx, (int)0, (int)25087)];
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
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)6143)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float val1 = in1_shared[((ci_lid + (32 * x1)) + (896 * x0))];
          float val2 = in2_shared[(co_tid + (33 * ci_lid))];
          int agg_idx = (x1_lid + (x0_lid * 7));
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
  for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
  {
    int x1 = ((4 * x1_lid) + x1_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T216 = agg[(x1_lid + (x0_lid * 7))];
      int gout_idx = (((co_gid + co_tid) + (5376 * (x0_gid + x0))) + (192 * x1));
      float LX_I_125 = X_I_125[(co_gid + co_tid)];
      float LX_I_124 = X_I_124[(co_gid + co_tid)];
      float LX_T217 = (LX_T216 - LX_I_125);
      float LX_T218 = (LX_T217 * LX_I_124);
      X_T218[gout_idx] = LX_T218;
    }
  }
}
