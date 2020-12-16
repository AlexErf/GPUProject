#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3328 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 792 }
// Out stride: { 1 }
// Elementwise input X_T7 shape: fp32(3, 3, 88, 1):(264, 88, 1, 1):3.09375 KiB
// Elementwise op: [[pid(RevMul)]] X_T8 = mul(X_T6, X_T7)
// Elementwise op: [[pid(Add)]] X_T9 = add(X_T5, X_T8)
// Tile size: { 64 }
// Contraction output var shape: fp32(3, 3, 88, 1):(264, 88, 1, 1):3.09375 KiB
// Computed true ops: 1584
// Computed work groups: 13
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 256, 1, 1
// gwork = 3328, 1, 1
__kernel void kernel_c29_sdk_1(__global float* restrict  X_T9, __global const float* restrict  X_T7)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 64);
  int i1_i2_i3_i4_tid = (tid % 64);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 768) || (i1_i2_i3_i4_tid < 24));
  if (i1_i2_i3_i4_cond)
  {
    if ((tid < 64))
    {
      int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
      float LX_T7 = X_T7[gout_idx];
      float LX_T8 = (0.1730969250202179f * LX_T7);
      float LX_T9 = (-0.08654846251010895f + LX_T8);
      X_T9[gout_idx] = LX_T9;
    }
  }
}
