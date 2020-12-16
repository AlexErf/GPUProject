#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 35 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 256 }
// Out stride: { 313600, 8960, 256, 1 }
// Elementwise input X_T899 shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Elementwise input X_T903 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_330 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T904 = div(X_T899, X_T903)
// Elementwise op: [[pid(Add, Switch)]] X_T905 = add(X_T904, X_I_330)
// Elementwise op: X_T906 = cmp_lt(X_T905, X_T2)
// Elementwise op: [[pid(Relu)]] X_T907 = cond(X_T906, X_T2, X_T905)
// Tile size: { 1, 1, 1, 256 }
// Contraction output var shape: fp32(1, 35, 35, 256):(313600, 8960, 256, 1):1225 KiB
// Computed true ops: 1254400
// Computed work groups: 1225
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 35, 1
__kernel void kernel_c51_sdk_294(__global float* restrict  X_T907, __global const float* restrict  X_T899, __global const float* restrict  X_T903, __global const float* restrict  X_I_330)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  int gout_idx = (((8960 * i2_gid) + (256 * i3_gid)) + i4_tid);
  float LX_T899 = X_T899[gout_idx];
  float LX_T903 = X_T903[i4_tid];
  float LX_I_330 = X_I_330[i4_tid];
  float LX_T904 = (LX_T899 / LX_T903);
  float LX_T905 = (LX_T904 + LX_I_330);
  int LX_T906 = (LX_T905 < 0.0f);
  float LX_T907 = select((float)LX_T905, (float)0.0f, (int)LX_T906);
  X_T907[gout_idx] = LX_T907;
}
