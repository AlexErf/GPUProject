#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 27648 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 442368 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 1, 384, 384):(147456, 147456, 384, 1):1728 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 4096 }
// Contraction output var shape: fp32(3, 1, 384, 384):(147456, 147456, 384, 1):1728 KiB
// Computed true ops: 884736
// Computed work groups: 108
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 512
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 27648, 1, 1
__kernel void kernel_c45_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 4096);
  int i1_i2_i3_i4_tid = (tid % 256);
  for (int i1_i2_i3_i4_lid = 0; i1_i2_i3_i4_lid < 16; i1_i2_i3_i4_lid += 1)
  {
    int i1_i2_i3_i4 = ((256 * i1_i2_i3_i4_lid) + i1_i2_i3_i4_tid);
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4);
    float LX_T7 = X_T7[gout_idx];
    float LX_T8 = (0.10206207633018494f * LX_T7);
    float LX_T9 = (-0.05103103816509247f + LX_T8);
    X_T9[gout_idx] = LX_T9;
  }
}
