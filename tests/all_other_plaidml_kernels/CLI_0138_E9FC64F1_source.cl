#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 353024 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 352800 }
// Out stride: { 1 }
// Elementwise input X_T186 shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Elementwise input X_T207 shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Elementwise op: X_T208 = add(X_T186, X_T207)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Computed true ops: 352800
// Computed work groups: 1379
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 353024, 1, 1
__kernel void kernel_c56_sdk_59(__global float* restrict  X_T208, __global const float* restrict  X_T186, __global const float* restrict  X_T207)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 352768) || (i2_i3_i4_tid < 32));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T186 = X_T186[gout_idx];
    float LX_T207 = X_T207[gout_idx];
    float LX_T208 = (LX_T186 + LX_T207);
    X_T208[gout_idx] = LX_T208;
  }
}
