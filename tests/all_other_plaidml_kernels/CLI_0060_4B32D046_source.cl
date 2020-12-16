#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 19200 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 153600 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(1, 1, 960, 160):(153600, 153600, 160, 1):600 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 2048 }
// Contraction output var shape: fp32(1, 1, 960, 160):(153600, 153600, 160, 1):600 KiB
// Computed true ops: 307200
// Computed work groups: 75
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 256
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 19200, 1, 1
__kernel void kernel_c36_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 2048);
  int i3_i4_tid = (tid % 256);
  for (int i3_i4_lid = 0; i3_i4_lid < 8; i3_i4_lid += 1)
  {
    int i3_i4 = ((256 * i3_i4_lid) + i3_i4_tid);
    int gout_idx = (i3_i4_gid + i3_i4);
    float LX_T7 = X_T7[gout_idx];
    float LX_T8 = (0.14638501405715942f * LX_T7);
    float LX_T9 = (-0.07319250702857971f + LX_T8);
    X_T9[gout_idx] = LX_T9;
  }
}
