#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 10752 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 172032 }
// Out stride: { 1 }
// Elementwise input X_T8 shape: fp32(1, 7, 128, 192):(172032, 24576, 192, 1):672 KiB
// Elementwise op: [[pid(RevMul)]] X_T9 = mul(X_T7, X_T8)
// Elementwise op: [[pid(Add)]] X_T10 = add(X_T6, X_T9)
// Tile size: { 4096 }
// Contraction output var shape: fp32(1, 7, 128, 192):(172032, 24576, 192, 1):672 KiB
// Computed true ops: 344064
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 512
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 10752, 1, 1
__kernel void kernel_c30_sdk_1(__global float* restrict  X_T10, __global const float* restrict  X_T8)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 4096);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 16; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T8 = X_T8[gout_idx];
    float LX_T9 = (0.10350983589887619f * LX_T8);
    float LX_T10 = (-0.051754917949438095f + LX_T9);
    X_T10[gout_idx] = LX_T10;
  }
}
