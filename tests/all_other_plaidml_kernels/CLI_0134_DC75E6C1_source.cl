#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 576 }
// Out stride: { 28224, 4032, 576, 1 }
// Elementwise input X_T571 shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Elementwise input X_I_24 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_23 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Sub)]] X_T572 = sub(X_T571, X_I_24)
// Elementwise op: [[pid(Mul)]] X_T573 = mul(X_T572, X_I_23)
// Tile size: { 1, 1, 1, 576 }
// Contraction output var shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Computed true ops: 56448
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 216
// Computed mem write: 2304
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_154(__global float* restrict  X_T573, __global const float* restrict  X_T571, __global const float* restrict  X_I_24, __global const float* restrict  X_I_23)
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
      float LX_T571 = X_T571[gout_idx];
      float LX_I_24 = X_I_24[i4];
      float LX_I_23 = X_I_23[i4];
      float LX_T572 = (LX_T571 - LX_I_24);
      float LX_T573 = (LX_T572 * LX_I_23);
      X_T573[gout_idx] = LX_T573;
    }
  }
}
