#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 11, 11, 672 }
// Out stride: { 81312, 7392, 672, 1 }
// Elementwise input X_T3036 shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Elementwise input X_T3048 shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Elementwise input X_I_1155 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_1154 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T3049 = add(X_T3036, X_T3048)
// Elementwise op: [[pid(Sub)]] X_T3050 = sub(X_T3049, X_I_1155)
// Elementwise op: [[pid(Mul)]] X_T3051 = mul(X_T3050, X_I_1154)
// Tile size: { 1, 4, 1, 64 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 243936
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_1184(__global float* restrict  X_T3051, __global const float* restrict  X_T3036, __global const float* restrict  X_T3048, __global const float* restrict  X_I_1155, __global const float* restrict  X_I_1154)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 64);
  int i3_gid = get_group_id(2);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = ((i4_gid != 640) || (i4_tid < 32));
  if (i4_cond)
  {
    int i2_cond = ((i2_gid != 8) || (i2_tid < 3));
    if (i2_cond)
    {
      int gout_idx = (((7392 * (i2_gid + i2_tid)) + (672 * i3_gid)) + (i4_gid + i4_tid));
      float LX_T3036 = X_T3036[gout_idx];
      float LX_T3048 = X_T3048[gout_idx];
      float LX_I_1155 = X_I_1155[(i4_gid + i4_tid)];
      float LX_I_1154 = X_I_1154[(i4_gid + i4_tid)];
      float LX_T3049 = (LX_T3036 + LX_T3048);
      float LX_T3050 = (LX_T3049 - LX_I_1155);
      float LX_T3051 = (LX_T3050 * LX_I_1154);
      X_T3051[gout_idx] = LX_T3051;
    }
  }
}
