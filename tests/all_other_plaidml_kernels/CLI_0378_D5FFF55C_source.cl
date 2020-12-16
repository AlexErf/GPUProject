#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 576 }
// Out stride: { 28224, 4032, 576, 1 }
// Elementwise input X_T1198 shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Elementwise input X_T1221 shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Elementwise input X_I_473 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_472 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1222 = add(X_T1198, X_T1221)
// Elementwise op: [[pid(Sub)]] X_T1224 = sub(X_T1222, X_I_473)
// Elementwise op: [[pid(Mul)]] X_T1225 = mul(X_T1224, X_I_472)
// Tile size: { 1, 1, 1, 576 }
// Contraction output var shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Computed true ops: 84672
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 288
// Computed mem write: 4608
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_419(__global float* restrict  X_T1222, __global float* restrict  X_T1225, __global const float* restrict  X_T1198, __global const float* restrict  X_T1221, __global const float* restrict  X_I_473, __global const float* restrict  X_I_472)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4032 * i2_gid) + (576 * i3_gid)) + i4);
      float LX_T1198 = X_T1198[gout_idx];
      float LX_T1221 = X_T1221[gout_idx];
      float LX_I_473 = X_I_473[i4];
      float LX_I_472 = X_I_472[i4];
      float LX_T1222 = (LX_T1198 + LX_T1221);
      float LX_T1224 = (LX_T1222 - LX_I_473);
      float LX_T1225 = (LX_T1224 * LX_I_472);
      X_T1222[gout_idx] = LX_T1222;
      X_T1225[gout_idx] = LX_T1225;
    }
  }
}
