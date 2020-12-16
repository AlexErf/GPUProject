#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1_i2 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_T6 shape: fp32(128, 1):(1, 1):512 bytes
// Elementwise op: [[pid(RevMul)]] X_T7 = mul(X_T5, X_T6)
// Elementwise op: [[pid(Add)]] X_T8 = add(X_T4, X_T7)
// Tile size: { 128 }
// Contraction output var shape: fp32(128, 1):(1, 1):512 bytes
// Computed true ops: 256
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c4_sdk_1(__global float* restrict  X_T8, __global const float* restrict  X_T6)
{
  int tid = get_local_id(0);
  int i1_i2_tid = (tid % 128);
  float LX_T6 = X_T6[i1_i2_tid];
  float LX_T7 = (0.4313310980796814f * LX_T6);
  float LX_T8 = (-0.2156655490398407f + LX_T7);
  X_T8[i1_i2_tid] = LX_T8;
}
