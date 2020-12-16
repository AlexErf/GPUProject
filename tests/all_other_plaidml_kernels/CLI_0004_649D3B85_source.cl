#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4608 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 36864 }
// Out stride: { 1 }
// Elementwise input X_T6 shape: fp32(3, 3, 64, 64):(12288, 4096, 64, 1):144 KiB
// Elementwise op: [[pid(RevMul)]] X_T7 = mul(X_T5, X_T6)
// Elementwise op: [[pid(Add)]] X_T8 = add(X_T4, X_T7)
// Tile size: { 2048 }
// Contraction output var shape: fp32(3, 3, 64, 64):(12288, 4096, 64, 1):144 KiB
// Computed true ops: 73728
// Computed work groups: 18
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 256
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4608, 1, 1
__kernel void kernel_c3_sdk_1(__global float* restrict  X_T8, __global const float* restrict  X_T6)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 2048);
  int i1_i2_i3_i4_tid = (tid % 256);
  for (int i1_i2_i3_i4_lid = 0; i1_i2_i3_i4_lid < 8; i1_i2_i3_i4_lid += 1)
  {
    int i1_i2_i3_i4 = ((256 * i1_i2_i3_i4_lid) + i1_i2_i3_i4_tid);
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4);
    float LX_T6 = X_T6[gout_idx];
    float LX_T7 = (0.14433756470680237f * LX_T6);
    float LX_T8 = (-0.07216878235340118f + LX_T7);
    X_T8[gout_idx] = LX_T8;
  }
}
