#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1824 }
// Out stride: { 89376, 12768, 1824, 1 }
// Elementwise input X_T2501 shape: fp32(1, 7, 7, 1824):(89376, 12768, 1824, 1):349.125 KiB
// Elementwise input X_T2524 shape: fp32(1, 7, 7, 1824):(89376, 12768, 1824, 1):349.125 KiB
// Elementwise input X_I_983 shape: fp32(1824):(1):7.125 KiB
// Elementwise input X_I_982 shape: fp32(1824):(1):7.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2525 = add(X_T2501, X_T2524)
// Elementwise op: [[pid(Sub)]] X_T2527 = sub(X_T2525, X_I_983)
// Elementwise op: [[pid(Mul)]] X_T2528 = mul(X_T2527, X_I_982)
// Tile size: { 1, 1, 1, 1824 }
// Contraction output var shape: fp32(1, 7, 7, 1824):(89376, 12768, 1824, 1):349.125 KiB
// Computed true ops: 268128
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 912
// Computed mem write: 14592
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_878(__global float* restrict  X_T2525, __global float* restrict  X_T2528, __global const float* restrict  X_T2501, __global const float* restrict  X_T2524, __global const float* restrict  X_I_983, __global const float* restrict  X_I_982)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12768 * i2_gid) + (1824 * i3_gid)) + i4);
      float LX_T2501 = X_T2501[gout_idx];
      float LX_T2524 = X_T2524[gout_idx];
      float LX_I_983 = X_I_983[i4];
      float LX_I_982 = X_I_982[i4];
      float LX_T2525 = (LX_T2501 + LX_T2524);
      float LX_T2527 = (LX_T2525 - LX_I_983);
      float LX_T2528 = (LX_T2527 * LX_I_982);
      X_T2525[gout_idx] = LX_T2525;
      X_T2528[gout_idx] = LX_T2528;
    }
  }
}
