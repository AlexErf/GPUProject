#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1856 }
// Out stride: { 90944, 12992, 1856, 1 }
// Elementwise input X_T2526 shape: fp32(1, 7, 7, 1856):(90944, 12992, 1856, 1):355.25 KiB
// Elementwise input X_T2549 shape: fp32(1, 7, 7, 1856):(90944, 12992, 1856, 1):355.25 KiB
// Elementwise input X_I_993 shape: fp32(1856):(1):7.25 KiB
// Elementwise input X_I_992 shape: fp32(1856):(1):7.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2550 = add(X_T2526, X_T2549)
// Elementwise op: [[pid(Sub)]] X_T2552 = sub(X_T2550, X_I_993)
// Elementwise op: [[pid(Mul)]] X_T2553 = mul(X_T2552, X_I_992)
// Tile size: { 1, 1, 1, 1856 }
// Contraction output var shape: fp32(1, 7, 7, 1856):(90944, 12992, 1856, 1):355.25 KiB
// Computed true ops: 272832
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 928
// Computed mem write: 14848
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_887(__global float* restrict  X_T2550, __global float* restrict  X_T2553, __global const float* restrict  X_T2526, __global const float* restrict  X_T2549, __global const float* restrict  X_I_993, __global const float* restrict  X_I_992)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12992 * i2_gid) + (1856 * i3_gid)) + i4);
      float LX_T2526 = X_T2526[gout_idx];
      float LX_T2549 = X_T2549[gout_idx];
      float LX_I_993 = X_I_993[i4];
      float LX_I_992 = X_I_992[i4];
      float LX_T2550 = (LX_T2526 + LX_T2549);
      float LX_T2552 = (LX_T2550 - LX_I_993);
      float LX_T2553 = (LX_T2552 * LX_I_992);
      X_T2550[gout_idx] = LX_T2550;
      X_T2553[gout_idx] = LX_T2553;
    }
  }
}
