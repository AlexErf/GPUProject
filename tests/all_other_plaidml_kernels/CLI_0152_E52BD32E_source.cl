#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 55552 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 110976 }
// Out stride: { 1 }
// Elementwise input X_T944 shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Elementwise input X_T980 shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T981 = add(X_T944, X_T980)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Computed true ops: 110976
// Computed work groups: 217
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 55552, 1, 1
__kernel void kernel_c51_sdk_320(__global float* restrict  X_T981, __global const float* restrict  X_T944, __global const float* restrict  X_T980)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 110592) || (i2_i3_i4_tid < 128)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T944 = X_T944[gout_idx];
      float LX_T980 = X_T980[gout_idx];
      float LX_T981 = (LX_T944 + LX_T980);
      X_T981[gout_idx] = LX_T981;
    }
  }
}
