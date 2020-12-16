#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16640 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 133120 }
// Out stride: { 1 }
// Elementwise input X_T1939 shape: fp32(1, 8, 8, 2080):(133120, 16640, 2080, 1):520 KiB
// Elementwise input X_T1960 shape: fp32(1, 8, 8, 2080):(133120, 16640, 2080, 1):520 KiB
// Elementwise op: X_T1961 = add(X_T1939, X_T1960)
// Tile size: { 2048 }
// Contraction output var shape: fp32(1, 8, 8, 2080):(133120, 16640, 2080, 1):520 KiB
// Computed true ops: 133120
// Computed work groups: 65
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 512
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 16640, 1, 1
__kernel void kernel_c51_sdk_640(__global float* restrict  X_T1961, __global const float* restrict  X_T1939, __global const float* restrict  X_T1960)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 2048);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 8; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T1939 = X_T1939[gout_idx];
    float LX_T1960 = X_T1960[gout_idx];
    float LX_T1961 = (LX_T1939 + LX_T1960);
    X_T1961[gout_idx] = LX_T1961;
  }
}
