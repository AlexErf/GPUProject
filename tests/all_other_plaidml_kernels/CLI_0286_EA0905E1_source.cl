#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 4 1
// lid: 256 1 1
// Original:
// X_T584[n, x0, x1, co : _T774, _T775, _T776, _T777] = +(X_T583[n, k0 + x0, k1 + x1, ci] * X_I_209[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T584[n, x0, x1, co : _T774, _T775, _T776, _T777] = +(X_T583[n, k0 + x0, k1 + x1, ci] * X_I_209[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 128, 0 <= co < 128, 0 <= ci < 256, 0 <= ci < 256, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + x0 < 14, 0 <= k1 + x1 < 14, 0 <= co < 128, 0 <= ci < 256 }
// Defracted:
// X_T584[n, x0, x1, co : _T774, _T775, _T776, _T777] = +(X_T583[n, k0 + x0, k1 + x1, ci] * X_I_209[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T584    X_T583   X_I_209  
//       ci       256         0         1       128  
//       co       128         1         0         1  
//       x0        14      1792      3584         0  
//       x1        14       128       256         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 256, 128, 14, 14 }
// Out stride: { 0, 1, 1792, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 3584, 256 }
// Input 2 offset: 0
// Input 2 stride: { 128, 1, 0, 0 }
// Elementwise input X_I_208 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_207 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Sub)]] X_T585 = sub(X_T584, X_I_208)
// Elementwise op: [[pid(Mul)]] X_T586 = mul(X_T585, X_I_207)
// Tile size: { 64, 32, 14, 4 }
// Contraction output var shape: fp32(1, 14, 14, 128):(25088, 1792, 128, 1):98 KiB
// Computed true ops: 25690112
// Computed work groups: 16
// Computed inner loops: 4
// Computed shared mem: 22896
// Computed out regs: 7168
// Computed mem read: 22976
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 4, 1
__kernel void kernel_c108_sdk_183(__global float* restrict  X_T586, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_208, __global const float* restrict  X_I_207)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3644];
  __local float in2_shared[2080];
  int co_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 4);
  for (int ci_gid = 0; ci_gid < 256; ci_gid += 64)
  {
    {
      int gbase = (ci_gid + (x1_gid * 256));
      int ci_tid = (tid % 64);
      int x1_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int lidx = ((ci_tid + (911 * x1_tid)) + (65 * x0_lid));
        int gidx = (((gbase + ci_tid) + (256 * x1_tid)) + (3584 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)50175)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 128));
      int co_tid = (tid % 32);
      int ci_tid = ((tid / 32) % 8);
      for (int ci_lid = 0; ci_lid < 8; ci_lid += 1)
      {
        int ci = ((8 * ci_lid) + ci_tid);
        int lidx = ((65 * co_tid) + ci);
        int gidx = ((gbase + co_tid) + (128 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)32767)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 64; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float val1 = in1_shared[((ci_lid + (911 * x1_tid)) + (65 * x0))];
        float val2 = in2_shared[((65 * co_tid) + ci_lid)];
        float agg_rhs = mad(val2, val1, agg[x0_lid]);
        agg[x0_lid] = agg_rhs;
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
    for (int x0_lid = 0; x0_lid < 7; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T584 = agg[x0_lid];
      int gout_idx = (((co_gid + co_tid) + (1792 * x0)) + (128 * (x1_gid + x1_tid)));
      float LX_I_208 = X_I_208[(co_gid + co_tid)];
      float LX_I_207 = X_I_207[(co_gid + co_tid)];
      float LX_T585 = (LX_T584 - LX_I_208);
      float LX_T586 = (LX_T585 * LX_I_207);
      X_T586[gout_idx] = LX_T586;
    }
  }
}
