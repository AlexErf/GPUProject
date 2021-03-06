#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5120 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 81920 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(1, 1, 640, 128):(81920, 81920, 128, 1):320 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 4096 }
// Contraction output var shape: fp32(1, 1, 640, 128):(81920, 81920, 128, 1):320 KiB
// Computed true ops: 163840
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 512
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5120, 1, 1
__kernel void kernel_c41_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 4096);
  int i3_i4_tid = (tid % 256);
  for (int i3_i4_lid = 0; i3_i4_lid < 16; i3_i4_lid += 1)
  {
    int i3_i4 = ((256 * i3_i4_lid) + i3_i4_tid);
    int gout_idx = (i3_i4_gid + i3_i4);
    float LX_T7 = X_T7[gout_idx];
    float LX_T8 = (0.1767766922712326f * LX_T7);
    float LX_T9 = (-0.0883883461356163f + LX_T8);
    X_T9[gout_idx] = LX_T9;
  }
}
