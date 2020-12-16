#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 4
// lid: 256 1 1
// Original:
// X_T213[n, x0, x1, co : _T214, _T215, _T216, _T217] = +(X_T212[n, k0 + x0, k1 + x1, ci] * X_I_70[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T213[n, x0, x1, co : _T214, _T215, _T216, _T217] = +(X_T212[n, k0 + x0, k1 + x1, ci] * X_I_70[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 256, 0 <= ci < 256, 0 <= ci < 256, 0 <= co < 256, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + x0 < 28, 0 <= k1 + x1 < 28, 0 <= co < 256, 0 <= ci < 256 }
// Defracted:
// X_T213[n, x0, x1, co : _T214, _T215, _T216, _T217] = +(X_T212[n, k0 + x0, k1 + x1, ci] * X_I_70[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T213    X_T212    X_I_70  
//       ci       256         0         1       256  
//       co       256         1         0         1  
//       x0        28      7168      7168         0  
//       x1        28       256       256         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co, x0, x1 }
// Ranges: { 256, 256, 28, 28 }
// Out stride: { 0, 1, 7168, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0, 7168, 256 }
// Input 2 offset: 0
// Input 2 stride: { 256, 1, 0, 0 }
// Elementwise input X_I_69 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_68 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Sub)]] X_T214 = sub(X_T213, X_I_69)
// Elementwise op: [[pid(Mul)]] X_T215 = mul(X_T214, X_I_68)
// Tile size: { 32, 64, 4, 16 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 205520896
// Computed work groups: 56
// Computed inner loops: 8
// Computed shared mem: 16784
// Computed out regs: 16384
// Computed mem read: 17408
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 4
__kernel void kernel_c25_sdk_52(__global float* restrict  X_T215, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_69, __global const float* restrict  X_I_68)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2116];
  __local float in2_shared[2080];
  int co_gid = (get_group_id(2) * 64);
  int x1_gid = (get_group_id(1) * 16);
  int x0_gid = (get_group_id(0) * 4);
  for (int ci_gid = 0; ci_gid < 256; ci_gid += 32)
  {
    {
      int gbase = ((ci_gid + (x1_gid * 256)) + (x0_gid * 7168));
      int ci_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 8);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1 = ((8 * x1_lid) + x1_tid);
        for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
        {
          int lidx = ((ci_tid + (33 * x1)) + (529 * x0_lid));
          int gidx = (((gbase + ci_tid) + (256 * x1)) + (7168 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
        }
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 256));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 8; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (256 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)65535)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int ci_lid = 0; ci_lid < 32; ci_lid += 1)
    {
      int co_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 4);
      int x0_tid = ((tid / 128) % 2);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        for (int co_lid = 0; co_lid < 2; co_lid += 1)
        {
          int co = ((32 * co_lid) + co_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((ci_lid + (33 * x1)) + (529 * x0))];
            float val2 = in2_shared[(co + (65 * ci_lid))];
            int agg_idx = ((co_lid + (x1_lid * 2)) + (x0_lid * 8));
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
  for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
  {
    int x1_cond = ((x1_lid < 3) || (x1_gid != 16));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int co_lid = 0; co_lid < 2; co_lid += 1)
      {
        int co = ((32 * co_lid) + co_tid);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          float LX_T213 = agg[((co_lid + (x1_lid * 2)) + (x0_lid * 8))];
          int gout_idx = (((co_gid + co) + (7168 * (x0_gid + x0))) + (256 * (x1_gid + x1)));
          float LX_I_69 = X_I_69[(co_gid + co)];
          float LX_I_68 = X_I_68[(co_gid + co)];
          float LX_T214 = (LX_T213 - LX_I_69);
          float LX_T215 = (LX_T214 * LX_I_68);
          X_T215[gout_idx] = LX_T215;
        }
      }
    }
  }
}
