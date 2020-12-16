#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 221952 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 221952 }
// Out stride: { 1 }
// Elementwise input X_T348 shape: fp32(1, 17, 17, 768):(221952, 13056, 768, 1):867 KiB
// Elementwise input X_T379 shape: fp32(1, 17, 17, 768):(221952, 13056, 768, 1):867 KiB
// Elementwise op: X_T380 = add(X_T348, X_T379)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 17, 17, 768):(221952, 13056, 768, 1):867 KiB
// Computed true ops: 221952
// Computed work groups: 867
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 221952, 1, 1
__kernel void kernel_c56_sdk_124(__global float* restrict  X_T380, __global const float* restrict  X_T348, __global const float* restrict  X_T379)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
  float LX_T348 = X_T348[gout_idx];
  float LX_T379 = X_T379[gout_idx];
  float LX_T380 = (LX_T348 + LX_T379);
  X_T380[gout_idx] = LX_T380;
}
