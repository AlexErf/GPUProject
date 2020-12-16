#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i3_i4 }
// Ranges: { 121 }
// Out stride: { 1 }
// Elementwise input X_T6 shape: fp32(1, 1, 11, 11):(121, 121, 11, 1):484 bytes
// Elementwise op: [[pid(RevMul)]] X_T7 = mul(X_T5, X_T6)
// Elementwise op: [[pid(Add)]] X_T8 = add(X_T4, X_T7)
// Tile size: { 121 }
// Contraction output var shape: fp32(1, 1, 11, 11):(121, 121, 11, 1):484 bytes
// Computed true ops: 242
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 121
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_1(__global float* restrict  X_T8, __global const float* restrict  X_T6)
{
  int tid = get_local_id(0);
  int i3_i4_tid = (tid % 128);
  int i3_i4_cond = (i3_i4_tid < 121);
  if (i3_i4_cond)
  {
    float LX_T6 = X_T6[i3_i4_tid];
    float LX_T7 = (1.0444658994674683f * LX_T6);
    float LX_T8 = (-0.5222329497337341f + LX_T7);
    X_T8[i3_i4_tid] = LX_T8;
  }
}
