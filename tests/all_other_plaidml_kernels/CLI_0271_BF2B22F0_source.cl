#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 51968 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 51744 }
// Out stride: { 1 }
// Elementwise input X_T2371 shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Elementwise input X_T2442 shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Elementwise op: X_T2443 = add(X_T2371, X_T2442)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Computed true ops: 51744
// Computed work groups: 203
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 51968, 1, 1
__kernel void kernel_c42_sdk_936(__global float* restrict  X_T2443, __global const float* restrict  X_T2371, __global const float* restrict  X_T2442)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 51712) || (i2_i3_i4_tid < 32));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T2371 = X_T2371[gout_idx];
    float LX_T2442 = X_T2442[gout_idx];
    float LX_T2443 = (LX_T2371 + LX_T2442);
    X_T2443[gout_idx] = LX_T2443;
  }
}
