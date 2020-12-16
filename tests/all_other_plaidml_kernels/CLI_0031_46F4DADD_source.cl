#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 6912 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 13824 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 32, 48):(4608, 1536, 48, 1):54 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 512 }
// Contraction output var shape: fp32(3, 3, 32, 48):(4608, 1536, 48, 1):54 KiB
// Computed true ops: 27648
// Computed work groups: 27
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 64
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 6912, 1, 1
__kernel void kernel_c19_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 512);
  int i1_i2_i3_i4_tid = (tid % 256);
  for (int i1_i2_i3_i4_lid = 0; i1_i2_i3_i4_lid < 2; i1_i2_i3_i4_lid += 1)
  {
    int i1_i2_i3_i4 = ((256 * i1_i2_i3_i4_lid) + i1_i2_i3_i4_tid);
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4);
    float LX_T7 = X_T7[gout_idx];
    float LX_T8 = (0.18257418274879456f * LX_T7);
    float LX_T9 = (-0.09128709137439728f + LX_T8);
    X_T9[gout_idx] = LX_T9;
  }
}
