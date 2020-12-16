#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 34560 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 68992 }
// Out stride: { 1 }
// Elementwise input X_T315 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise input X_T353 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise op: X_T354 = add(X_T315, X_T353)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Computed true ops: 68992
// Computed work groups: 135
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 34560, 1, 1
__kernel void kernel_c42_sdk_121(__global float* restrict  X_T354, __global const float* restrict  X_T315, __global const float* restrict  X_T353)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 68608) || (i2_i3_i4_tid < 128)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T315 = X_T315[gout_idx];
      float LX_T353 = X_T353[gout_idx];
      float LX_T354 = (LX_T315 + LX_T353);
      X_T354[gout_idx] = LX_T354;
    }
  }
}
