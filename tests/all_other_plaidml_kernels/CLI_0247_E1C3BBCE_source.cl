#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 325376 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 325248 }
// Out stride: { 1 }
// Elementwise input X_T2878 shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Elementwise input X_T2915 shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Elementwise op: X_T2916 = add(X_T2878, X_T2915)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Computed true ops: 325248
// Computed work groups: 1271
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 325376, 1, 1
__kernel void kernel_c42_sdk_1127(__global float* restrict  X_T2916, __global const float* restrict  X_T2878, __global const float* restrict  X_T2915)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 325120) || (i2_i3_i4_tid < 128));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T2878 = X_T2878[gout_idx];
    float LX_T2915 = X_T2915[gout_idx];
    float LX_T2916 = (LX_T2878 + LX_T2915);
    X_T2916[gout_idx] = LX_T2916;
  }
}
