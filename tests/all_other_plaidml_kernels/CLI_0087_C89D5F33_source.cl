#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 137984 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 137984 }
// Out stride: { 1 }
// Elementwise input X_T95 shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Elementwise input X_T136 shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Elementwise op: X_T137 = add(X_T95, X_T136)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Computed true ops: 137984
// Computed work groups: 539
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 137984, 1, 1
__kernel void kernel_c42_sdk_35(__global float* restrict  X_T137, __global const float* restrict  X_T95, __global const float* restrict  X_T136)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
  float LX_T95 = X_T95[gout_idx];
  float LX_T136 = X_T136[gout_idx];
  float LX_T137 = (LX_T95 + LX_T136);
  X_T137[gout_idx] = LX_T137;
}
