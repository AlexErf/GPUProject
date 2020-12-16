#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 24 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 768 }
// Out stride: { 150528, 10752, 768, 1 }
// Elementwise input X_T948 shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Elementwise input X_T971 shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Elementwise input X_I_372 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_371 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(Concatenate)]] X_T972 = add(X_T948, X_T971)
// Elementwise op: [[pid(Sub)]] X_T974 = sub(X_T972, X_I_372)
// Elementwise op: [[pid(Mul)]] X_T975 = mul(X_T974, X_I_371)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Computed true ops: 451584
// Computed work groups: 168
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 24, 1
__kernel void kernel_c108_sdk_323(__global float* restrict  X_T972, __global float* restrict  X_T975, __global const float* restrict  X_T948, __global const float* restrict  X_T971, __global const float* restrict  X_I_372, __global const float* restrict  X_I_371)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((10752 * (i2_gid + i2_tid)) + (768 * i3)) + (i4_gid + i4_tid));
      float LX_T948 = X_T948[gout_idx];
      float LX_T971 = X_T971[gout_idx];
      float LX_I_372 = X_I_372[(i4_gid + i4_tid)];
      float LX_I_371 = X_I_371[(i4_gid + i4_tid)];
      float LX_T972 = (LX_T948 + LX_T971);
      float LX_T974 = (LX_T972 - LX_I_372);
      float LX_T975 = (LX_T974 * LX_I_371);
      X_T972[gout_idx] = LX_T972;
      X_T975[gout_idx] = LX_T975;
    }
  }
}
