#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 2
// lid: 256 1 1
// Original:
// X_T198[n, x0, x1, co : _T230, _T231, _T232, _T233] = +(X_T197[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T198[n, x0, x1, co : _T230, _T231, _T232, _T233] = +(X_T197[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= co < 128, 0 <= co < 128, 0 <= ci < 224, 0 <= ci < 224, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + x0 < 56, 0 <= k1 + x1 < 56, 0 <= co < 128, 0 <= ci < 224 }
// Defracted:
// X_T198[n, x0, x1, co : _T230, _T231, _T232, _T233] = +(X_T197[n, k0 + x0, k1 + x1, ci] * X_I_77[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T198    X_T197    X_I_77  
//       ci       224         0         1       128  
//       co       128         1         0         1  
//       x0        56      7168     12544         0  
//       x1        56       128       224         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 224, 128, 56, 56 }
// Out stride: { 0, 1, 7168, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 12544, 224 }
// Input 2 offset: 0
// Input 2 stride: { 128, 1, 0, 0 }
// Elementwise input X_I_76 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_75 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Sub)]] X_T199 = sub(X_T198, X_I_76)
// Elementwise op: [[pid(Mul)]] X_T200 = mul(X_T199, X_I_75)
// Tile size: { 32, 64, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 359661568
// Computed work groups: 98
// Computed inner loops: 7
// Computed shared mem: 16800
// Computed out regs: 16384
// Computed mem read: 17408
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 2
__kernel void kernel_c68_sdk_54(__global float* restrict  X_T200, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_76, __global const float* restrict  X_I_75)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2120];
  __local float in2_shared[2080];
  int co_gid = (get_group_id(2) * 64);
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int ci_gid = 0; ci_gid < 224; ci_gid += 32)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 224)) + (x0_gid * 12544));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 8);
      for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
      {
        int lidx = ((ci_tid + (33 * x1_tid)) + (265 * x0_lid));
        int gidx = (((gbase + ci_tid) + (224 * x1_tid)) + (12544 * x0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)702463)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 128));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 8; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (128 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)28671)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (33 * x1)) + (265 * x0))];
            float val2 = in2_shared[(co + (65 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 4));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = agg_rhs;
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int co_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int co_lid = 0; co_lid < 2; co_lid += 1)
  {
    int co = ((32 * co_lid) + co_tid);
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        float LX_T198 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 4))];
        int gout_idx = (((co_gid + co) + (7168 * (x0_gid + x0))) + (128 * (x1_gid + x1)));
        float LX_I_76 = X_I_76[(co_gid + co)];
        float LX_I_75 = X_I_75[(co_gid + co)];
        float LX_T199 = (LX_T198 - LX_I_76);
        float LX_T200 = (LX_T199 * LX_I_75);
        X_T200[gout_idx] = LX_T200;
      }
    }
  }
}
