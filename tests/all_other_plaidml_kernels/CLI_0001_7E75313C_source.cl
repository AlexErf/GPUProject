#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 864 }
// Out stride: { 1 }
// Elementwise input X_T6 shape: fp32(3, 3, 3, 32):(288, 96, 32, 1):3.375 KiB
// Elementwise op: [[pid(RevMul)]] X_T7 = mul(X_T5, X_T6)
// Elementwise op: [[pid(Add)]] X_T8 = add(X_T4, X_T7)
// Tile size: { 256 }
// Contraction output var shape: fp32(3, 3, 3, 32):(288, 96, 32, 1):3.375 KiB
// Computed true ops: 1728
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c1_sdk_1(__global float* restrict  X_T8, __global const float* restrict  X_T6)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 256);
  int i1_i2_i3_i4_tid = (tid % 256);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 768) || (i1_i2_i3_i4_tid < 96));
  if (i1_i2_i3_i4_cond)
  {
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
    float LX_T6 = X_T6[gout_idx];
    float LX_T7 = (0.2760262191295624f * LX_T6);
    float LX_T8 = (-0.1380131095647812f + LX_T7);
    X_T8[gout_idx] = LX_T8;
  }
}
