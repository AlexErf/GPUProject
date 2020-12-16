#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 608 }
// Out stride: { 29792, 4256, 608, 1 }
// Elementwise input X_T1223 shape: fp32(1, 7, 7, 608):(29792, 4256, 608, 1):116.375 KiB
// Elementwise input X_T1246 shape: fp32(1, 7, 7, 608):(29792, 4256, 608, 1):116.375 KiB
// Elementwise input X_I_483 shape: fp32(608):(1):2.375 KiB
// Elementwise input X_I_482 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1247 = add(X_T1223, X_T1246)
// Elementwise op: [[pid(Sub)]] X_T1249 = sub(X_T1247, X_I_483)
// Elementwise op: [[pid(Mul)]] X_T1250 = mul(X_T1249, X_I_482)
// Tile size: { 1, 1, 1, 608 }
// Contraction output var shape: fp32(1, 7, 7, 608):(29792, 4256, 608, 1):116.375 KiB
// Computed true ops: 89376
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 304
// Computed mem write: 4864
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_428(__global float* restrict  X_T1247, __global float* restrict  X_T1250, __global const float* restrict  X_T1223, __global const float* restrict  X_T1246, __global const float* restrict  X_I_483, __global const float* restrict  X_I_482)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4256 * i2_gid) + (608 * i3_gid)) + i4);
      float LX_T1223 = X_T1223[gout_idx];
      float LX_T1246 = X_T1246[gout_idx];
      float LX_I_483 = X_I_483[i4];
      float LX_I_482 = X_I_482[i4];
      float LX_T1247 = (LX_T1223 + LX_T1246);
      float LX_T1249 = (LX_T1247 - LX_I_483);
      float LX_T1250 = (LX_T1249 * LX_I_482);
      X_T1247[gout_idx] = LX_T1247;
      X_T1250[gout_idx] = LX_T1250;
    }
  }
}
