#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 64 }
// Out stride: { 12544, 896, 64, 1 }
// Elementwise input X_T322 shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Elementwise input X_T326 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_42 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T327 = div(X_T322, X_T326)
// Elementwise op: [[pid(Add, Switch)]] X_T328 = add(X_T327, X_I_42)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Computed true ops: 25088
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_83(__global float* restrict  X_T328, __global const float* restrict  X_T322, __global const float* restrict  X_T326, __global const float* restrict  X_I_42)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int gout_idx = (((896 * (i2_gid + i2_tid)) + (64 * (i3_gid + i3_tid))) + i4_tid);
  float LX_T322 = X_T322[gout_idx];
  float LX_T326 = X_T326[i4_tid];
  float LX_I_42 = X_I_42[i4_tid];
  float LX_T327 = (LX_T322 / LX_T326);
  float LX_T328 = (LX_T327 + LX_I_42);
  X_T328[gout_idx] = LX_T328;
}
