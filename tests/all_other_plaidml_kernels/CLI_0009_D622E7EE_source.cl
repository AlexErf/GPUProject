#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 1536 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(1, 1, 16, 96):(1536, 1536, 96, 1):6 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 1, 16, 96):(1536, 1536, 96, 1):6 KiB
// Computed true ops: 3072
// Computed work groups: 6
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 1, 1
__kernel void kernel_c6_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 256);
  int i3_i4_tid = (tid % 256);
  int gout_idx = (i3_i4_gid + i3_i4_tid);
  float LX_T7 = X_T7[gout_idx];
  float LX_T8 = (0.4629100561141968f * LX_T7);
  float LX_T9 = (-0.2314550280570984f + LX_T8);
  X_T9[gout_idx] = LX_T9;
}
