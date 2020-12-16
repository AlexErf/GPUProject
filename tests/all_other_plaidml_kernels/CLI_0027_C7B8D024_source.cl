#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 198 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 22, 1):(66, 22, 1, 1):792 bytes
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 198 }
// Contraction output var shape: fp32(3, 3, 22, 1):(66, 22, 1, 1):792 bytes
// Computed true ops: 396
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 28
// Computed mem write: 896
// Computed operations: 198
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c17_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_tid = (tid % 256);
  int i1_i2_i3_i4_cond = (i1_i2_i3_i4_tid < 198);
  if (i1_i2_i3_i4_cond)
  {
    float LX_T7 = X_T7[i1_i2_i3_i4_tid];
    float LX_T8 = (0.34050261974334717f * LX_T7);
    float LX_T9 = (-0.17025130987167358f + LX_T8);
    X_T9[i1_i2_i3_i4_tid] = LX_T9;
  }
}
