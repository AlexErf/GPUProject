#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 21 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 672 }
// Out stride: { 131712, 9408, 672, 1 }
// Elementwise input X_T853 shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Elementwise input X_T876 shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Elementwise input X_I_342 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_341 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T877 = add(X_T853, X_T876)
// Elementwise op: [[pid(Sub)]] X_T879 = sub(X_T877, X_I_342)
// Elementwise op: [[pid(Mul)]] X_T880 = mul(X_T879, X_I_341)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Computed true ops: 395136
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 21, 1
__kernel void kernel_c68_sdk_296(__global float* restrict  X_T877, __global float* restrict  X_T880, __global const float* restrict  X_T853, __global const float* restrict  X_T876, __global const float* restrict  X_I_342, __global const float* restrict  X_I_341)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((9408 * (i2_gid + i2_tid)) + (672 * i3)) + (i4_gid + i4_tid));
      float LX_T853 = X_T853[gout_idx];
      float LX_T876 = X_T876[gout_idx];
      float LX_I_342 = X_I_342[(i4_gid + i4_tid)];
      float LX_I_341 = X_I_341[(i4_gid + i4_tid)];
      float LX_T877 = (LX_T853 + LX_T876);
      float LX_T879 = (LX_T877 - LX_I_342);
      float LX_T880 = (LX_T879 * LX_I_341);
      X_T877[gout_idx] = LX_T877;
      X_T880[gout_idx] = LX_T880;
    }
  }
}
