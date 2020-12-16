#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 11
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 21, 21, 672 }
// Out stride: { 296352, 14112, 672, 1 }
// Elementwise input X_T2825 shape: fp32(1, 21, 21, 672):(296352, 14112, 672, 1):1157.62 KiB
// Elementwise input X_T2829 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_14 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2830 = div(X_T2825, X_T2829)
// Elementwise op: [[pid(Add, Switch)]] X_T2831 = add(X_T2830, X_I_14)
// Elementwise op: X_T2923 = cmp_lt(X_T2831, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2924 = cond(X_T2923, X_T1, X_T2831)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 21, 21, 672):(296352, 14112, 672, 1):1157.62 KiB
// Computed true ops: 1185408
// Computed work groups: 1331
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 11
__kernel void kernel_c42_sdk_1094(__global float* restrict  X_T2831, __global float* restrict  X_T2924, __global const float* restrict  X_T2825, __global const float* restrict  X_T2829, __global const float* restrict  X_I_14)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i4_cond = ((i4_gid != 640) || (i4_tid < 32));
  if (i4_cond)
  {
    int i3_cond = ((i3_gid != 20) || (i3_tid < 1));
    if (i3_cond)
    {
      int i2_cond = ((i2_gid != 20) || (i2_tid < 1));
      if (i2_cond)
      {
        int gout_idx = (((14112 * (i2_gid + i2_tid)) + (672 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
        float LX_T2825 = X_T2825[gout_idx];
        float LX_T2829 = X_T2829[(i4_gid + i4_tid)];
        float LX_I_14 = X_I_14[(i4_gid + i4_tid)];
        float LX_T2830 = (LX_T2825 / LX_T2829);
        float LX_T2831 = (LX_T2830 + LX_I_14);
        int LX_T2923 = (LX_T2831 < 0.0f);
        float LX_T2924 = select((float)LX_T2831, (float)0.0f, (int)LX_T2923);
        X_T2831[gout_idx] = LX_T2831;
        X_T2924[gout_idx] = LX_T2924;
      }
    }
  }
}
