#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 99 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 11, 1):(33, 11, 1, 1):396 bytes
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 99 }
// Contraction output var shape: fp32(3, 3, 11, 1):(33, 11, 1, 1):396 bytes
// Computed true ops: 198
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 99
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c11_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_tid = (tid % 128);
  int i1_i2_i3_i4_cond = (i1_i2_i3_i4_tid < 99);
  if (i1_i2_i3_i4_cond)
  {
    float LX_T7 = X_T7[i1_i2_i3_i4_tid];
    float LX_T8 = (0.4714045226573944f * LX_T7);
    float LX_T9 = (-0.2357022613286972f + LX_T8);
    X_T9[i1_i2_i3_i4_tid] = LX_T9;
  }
}
