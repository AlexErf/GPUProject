#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 137984 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 137984 }
// Out stride: { 1 }
// Elementwise input X_T204 shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Elementwise input X_T234 shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Elementwise op: [[pid(Concatenate)]] X_T235 = add(X_T204, X_T234)
// Elementwise op: X_T236 = cmp_lt(X_T235, X_T1)
// Elementwise op: [[pid(Relu)]] X_T237 = cond(X_T236, X_T1, X_T235)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Computed true ops: 413952
// Computed work groups: 539
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 137984, 1, 1
__kernel void kernel_c42_sdk_73(__global float* restrict  X_T237, __global const float* restrict  X_T204, __global const float* restrict  X_T234)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
  float LX_T204 = X_T204[gout_idx];
  float LX_T234 = X_T234[gout_idx];
  float LX_T235 = (LX_T204 + LX_T234);
  int LX_T236 = (LX_T235 < 0.0f);
  float LX_T237 = select((float)LX_T235, (float)0.0f, (int)LX_T236);
  X_T237[gout_idx] = LX_T237;
}
