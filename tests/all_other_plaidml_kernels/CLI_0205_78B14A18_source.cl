#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8192 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 131072 }
// Out stride: { 1 }
// Elementwise input X_T907 shape: fp32(1, 8, 8, 2048):(131072, 16384, 2048, 1):512 KiB
// Elementwise input X_T945 shape: fp32(1, 8, 8, 2048):(131072, 16384, 2048, 1):512 KiB
// Elementwise op: X_T946 = add(X_T907, X_T945)
// Tile size: { 4096 }
// Contraction output var shape: fp32(1, 8, 8, 2048):(131072, 16384, 2048, 1):512 KiB
// Computed true ops: 131072
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 1024
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8192, 1, 1
__kernel void kernel_c56_sdk_323(__global float* restrict  X_T946, __global const float* restrict  X_T907, __global const float* restrict  X_T945)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 4096);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 16; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T907 = X_T907[gout_idx];
    float LX_T945 = X_T945[gout_idx];
    float LX_T946 = (LX_T907 + LX_T945);
    X_T946[gout_idx] = LX_T946;
  }
}
