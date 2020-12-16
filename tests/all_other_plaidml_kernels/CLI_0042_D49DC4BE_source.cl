#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7936 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 7744 }
// Out stride: { 1 }
// Elementwise input X_T6 shape: fp32(1, 1, 88, 88):(7744, 7744, 88, 1):30.25 KiB
// Elementwise op: [[pid(RevMul)]] X_T7 = mul(X_T5, X_T6)
// Elementwise op: [[pid(Add)]] X_T8 = add(X_T4, X_T7)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 1, 88, 88):(7744, 7744, 88, 1):30.25 KiB
// Computed true ops: 15488
// Computed work groups: 31
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7936, 1, 1
__kernel void kernel_c27_sdk_1(__global float* restrict  X_T8, __global const float* restrict  X_T6)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 256);
  int i3_i4_tid = (tid % 256);
  int i3_i4_cond = ((i3_i4_gid != 7680) || (i3_i4_tid < 64));
  if (i3_i4_cond)
  {
    int gout_idx = (i3_i4_gid + i3_i4_tid);
    float LX_T6 = X_T6[gout_idx];
    float LX_T7 = (0.36927446722984314f * LX_T6);
    float LX_T8 = (-0.18463723361492157f + LX_T7);
    X_T8[gout_idx] = LX_T8;
  }
}
