#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 148224 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 592704 }
// Out stride: { 1 }
// Elementwise input X_T317 shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Elementwise input X_T355 shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Elementwise op: X_T356 = add(X_T317, X_T355)
// Tile size: { 1024 }
// Contraction output var shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Computed true ops: 592704
// Computed work groups: 579
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 148224, 1, 1
__kernel void kernel_c42_sdk_121(__global float* restrict  X_T356, __global const float* restrict  X_T317, __global const float* restrict  X_T355)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 1024);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 4; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 3) || ((i2_i3_i4_gid != 591872) || (i2_i3_i4_tid < 64)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T317 = X_T317[gout_idx];
      float LX_T355 = X_T355[gout_idx];
      float LX_T356 = (LX_T317 + LX_T355);
      X_T356[gout_idx] = LX_T356;
    }
  }
}
