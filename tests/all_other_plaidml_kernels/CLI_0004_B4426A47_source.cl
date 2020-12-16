#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 288 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 32, 1):(96, 32, 1, 1):1.125 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 128 }
// Contraction output var shape: fp32(3, 3, 32, 1):(96, 32, 1, 1):1.125 KiB
// Computed true ops: 576
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c3_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 128);
  int i1_i2_i3_i4_tid = (tid % 128);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 256) || (i1_i2_i3_i4_tid < 32));
  if (i1_i2_i3_i4_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
      float LX_T7 = X_T7[gout_idx];
      float LX_T8 = (0.2842676341533661f * LX_T7);
      float LX_T9 = (-0.14213381707668304f + LX_T8);
      X_T9[gout_idx] = LX_T9;
    }
  }
}
