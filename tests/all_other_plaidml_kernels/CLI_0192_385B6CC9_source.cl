#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5120 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 81920 }
// Out stride: { 1 }
// Elementwise input X_T851 shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Elementwise input X_T892 shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Elementwise op: X_T893 = add(X_T851, X_T892)
// Tile size: { 4096 }
// Contraction output var shape: fp32(1, 8, 8, 1280):(81920, 10240, 1280, 1):320 KiB
// Computed true ops: 81920
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 1024
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5120, 1, 1
__kernel void kernel_c56_sdk_302(__global float* restrict  X_T893, __global const float* restrict  X_T851, __global const float* restrict  X_T892)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 4096);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 16; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T851 = X_T851[gout_idx];
    float LX_T892 = X_T892[gout_idx];
    float LX_T893 = (LX_T851 + LX_T892);
    X_T893[gout_idx] = LX_T893;
  }
}
