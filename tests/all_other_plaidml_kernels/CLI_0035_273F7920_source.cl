#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3328 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 396 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 44, 1):(132, 44, 1, 1):1.54688 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 32 }
// Contraction output var shape: fp32(3, 3, 44, 1):(132, 44, 1, 1):1.54688 KiB
// Computed true ops: 792
// Computed work groups: 13
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 32
// Computed rollups: 0
// Computed threads used: 32
// lwork = 256, 1, 1
// gwork = 3328, 1, 1
__kernel void kernel_c22_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 32);
  int i1_i2_i3_i4_tid = (tid % 32);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 384) || (i1_i2_i3_i4_tid < 12));
  if (i1_i2_i3_i4_cond)
  {
    if ((tid < 32))
    {
      int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
      float LX_T7 = X_T7[gout_idx];
      float LX_T8 = (0.24343225359916687f * LX_T7);
      float LX_T9 = (-0.12171612679958344f + LX_T8);
      X_T9[gout_idx] = LX_T9;
    }
  }
}
