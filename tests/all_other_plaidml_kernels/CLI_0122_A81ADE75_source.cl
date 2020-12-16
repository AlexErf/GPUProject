#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 313600 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 313600 }
// Out stride: { 1 }
// Elementwise input X_T95 shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Elementwise input X_T121 shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Elementwise op: X_T122 = add(X_T95, X_T121)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Computed true ops: 313600
// Computed work groups: 1225
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 313600, 1, 1
__kernel void kernel_c56_sdk_28(__global float* restrict  X_T122, __global const float* restrict  X_T95, __global const float* restrict  X_T121)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
  float LX_T95 = X_T95[gout_idx];
  float LX_T121 = X_T121[gout_idx];
  float LX_T122 = (LX_T95 + LX_T121);
  X_T122[gout_idx] = LX_T122;
}
