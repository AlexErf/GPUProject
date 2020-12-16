#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 2400 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(5, 5, 96, 1):(480, 96, 1, 1):9.375 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 256 }
// Contraction output var shape: fp32(5, 5, 96, 1):(480, 96, 1, 1):9.375 KiB
// Computed true ops: 4800
// Computed work groups: 10
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 1, 1
__kernel void kernel_c10_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 256);
  int i1_i2_i3_i4_tid = (tid % 256);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 2304) || (i1_i2_i3_i4_tid < 96));
  if (i1_i2_i3_i4_cond)
  {
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
    float LX_T7 = X_T7[gout_idx];
    float LX_T8 = (0.0994831994175911f * LX_T7);
    float LX_T9 = (-0.04974159970879555f + LX_T8);
    X_T9[gout_idx] = LX_T9;
  }
}
