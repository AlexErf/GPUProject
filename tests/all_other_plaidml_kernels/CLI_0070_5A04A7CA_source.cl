#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16128 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 129024 }
// Out stride: { 1 }
// Elementwise input X_T8 shape: fp32(1, 3, 192, 224):(129024, 43008, 224, 1):504 KiB
// Elementwise op: [[pid(RevMul)]] X_T9 = mul(X_T7, X_T8)
// Elementwise op: [[pid(Add)]] X_T10 = add(X_T6, X_T9)
// Tile size: { 2048 }
// Contraction output var shape: fp32(1, 3, 192, 224):(129024, 43008, 224, 1):504 KiB
// Computed true ops: 258048
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 256
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 16128, 1, 1
__kernel void kernel_c42_sdk_1(__global float* restrict  X_T10, __global const float* restrict  X_T8)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 2048);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 8; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T8 = X_T8[gout_idx];
    float LX_T9 = (0.1386750489473343f * LX_T8);
    float LX_T10 = (-0.06933752447366714f + LX_T9);
    X_T10[gout_idx] = LX_T10;
  }
}
