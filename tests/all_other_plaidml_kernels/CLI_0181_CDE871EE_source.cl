#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 28672 }
// Out stride: { 1 }
// Elementwise input X_T2012 shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Elementwise input X_T2047 shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2048 = add(X_T2012, X_T2047)
// Tile size: { 1024 }
// Contraction output var shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Computed true ops: 28672
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c51_sdk_669(__global float* restrict  X_T2048, __global const float* restrict  X_T2012, __global const float* restrict  X_T2047)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 1024);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 4; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T2012 = X_T2012[gout_idx];
    float LX_T2047 = X_T2047[gout_idx];
    float LX_T2048 = (LX_T2012 + LX_T2047);
    X_T2048[gout_idx] = LX_T2048;
  }
}
