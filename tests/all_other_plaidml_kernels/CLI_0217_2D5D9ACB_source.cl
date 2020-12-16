#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 889088 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 889056 }
// Out stride: { 1 }
// Elementwise input X_T1790 shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Elementwise input X_T1861 shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Elementwise op: X_T1862 = add(X_T1790, X_T1861)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Computed true ops: 889056
// Computed work groups: 3473
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 889088, 1, 1
__kernel void kernel_c42_sdk_710(__global float* restrict  X_T1862, __global const float* restrict  X_T1790, __global const float* restrict  X_T1861)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 888832) || (i2_i3_i4_tid < 224));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T1790 = X_T1790[gout_idx];
    float LX_T1861 = X_T1861[gout_idx];
    float LX_T1862 = (LX_T1790 + LX_T1861);
    X_T1862[gout_idx] = LX_T1862;
  }
}
