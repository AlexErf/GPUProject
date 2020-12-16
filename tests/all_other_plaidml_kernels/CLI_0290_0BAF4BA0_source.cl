#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 288 }
// Out stride: { 56448, 4032, 288, 1 }
// Elementwise input X_T569 shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Elementwise input X_T596 shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Elementwise input X_I_222 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_221 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T597 = add(X_T569, X_T596)
// Elementwise op: [[pid(Sub)]] X_T599 = sub(X_T597, X_I_222)
// Elementwise op: [[pid(Mul)]] X_T600 = mul(X_T599, X_I_221)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Computed true ops: 169344
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c108_sdk_188(__global float* restrict  X_T597, __global float* restrict  X_T600, __global const float* restrict  X_T569, __global const float* restrict  X_T596, __global const float* restrict  X_I_222, __global const float* restrict  X_I_221)
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
      int gout_idx = (((4032 * (i2_gid + i2_tid)) + (288 * i3)) + (i4_gid + i4_tid));
      float LX_T569 = X_T569[gout_idx];
      float LX_T596 = X_T596[gout_idx];
      float LX_I_222 = X_I_222[(i4_gid + i4_tid)];
      float LX_I_221 = X_I_221[(i4_gid + i4_tid)];
      float LX_T597 = (LX_T569 + LX_T596);
      float LX_T599 = (LX_T597 - LX_I_222);
      float LX_T600 = (LX_T599 * LX_I_221);
      X_T597[gout_idx] = LX_T597;
      X_T600[gout_idx] = LX_T600;
    }
  }
}
