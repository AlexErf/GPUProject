#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 144 }
// Out stride: { 112896, 4032, 144, 1 }
// Elementwise input X_T189 shape: fp32(1, 28, 28, 144):(112896, 4032, 144, 1):441 KiB
// Elementwise input X_I_60 shape: fp32(144):(1):576 bytes
// Elementwise input X_I_59 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(Sub)]] X_T190 = sub(X_T189, X_I_60)
// Elementwise op: [[pid(Mul)]] X_T191 = mul(X_T190, X_I_59)
// Tile size: { 1, 4, 1, 144 }
// Contraction output var shape: fp32(1, 28, 28, 144):(112896, 4032, 144, 1):441 KiB
// Computed true ops: 225792
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 240
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c43_sdk_45(__global float* restrict  X_T191, __global const float* restrict  X_T189, __global const float* restrict  X_I_60, __global const float* restrict  X_I_59)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 16));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((4032 * (i2_gid + i2_tid)) + (144 * i3_gid)) + i4);
      float LX_T189 = X_T189[gout_idx];
      float LX_I_60 = X_I_60[i4];
      float LX_I_59 = X_I_59[i4];
      float LX_T190 = (LX_T189 - LX_I_60);
      float LX_T191 = (LX_T190 * LX_I_59);
      X_T191[gout_idx] = LX_T191;
    }
  }
}
