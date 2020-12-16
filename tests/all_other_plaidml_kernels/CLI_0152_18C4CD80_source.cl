#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 44 }
// Out stride: { 34496, 1232, 44, 1 }
// Elementwise input X_T488 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise input X_T500 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise input X_I_193 shape: fp32(44):(1):176 bytes
// Elementwise input X_I_192 shape: fp32(44):(1):176 bytes
// Elementwise op: [[pid(Concatenate)]] X_T501 = add(X_T488, X_T500)
// Elementwise op: [[pid(Sub)]] X_T502 = sub(X_T501, X_I_193)
// Elementwise op: [[pid(Mul)]] X_T503 = mul(X_T502, X_I_192)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 103488
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c42_sdk_178(__global float* restrict  X_T503, __global const float* restrict  X_T488, __global const float* restrict  X_T500, __global const float* restrict  X_I_193, __global const float* restrict  X_I_192)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = ((i4_gid != 32) || (i4_tid < 12));
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1232 * i2) + (44 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
      float LX_T488 = X_T488[gout_idx];
      float LX_T500 = X_T500[gout_idx];
      float LX_I_193 = X_I_193[(i4_gid + i4_tid)];
      float LX_I_192 = X_I_192[(i4_gid + i4_tid)];
      float LX_T501 = (LX_T488 + LX_T500);
      float LX_T502 = (LX_T501 - LX_I_193);
      float LX_T503 = (LX_T502 * LX_I_192);
      X_T503[gout_idx] = LX_T503;
    }
  }
}
