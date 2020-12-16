#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 288 }
// Out stride: { 56448, 4032, 288, 1 }
// Elementwise input X_T577 shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Elementwise input X_T604 shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Elementwise input X_I_222 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_221 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T605 = add(X_T577, X_T604)
// Elementwise op: [[pid(Sub)]] X_T607 = sub(X_T605, X_I_222)
// Elementwise op: [[pid(Mul)]] X_T608 = mul(X_T607, X_I_221)
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
__kernel void kernel_c124_sdk_188(__global float* restrict  X_T605, __global float* restrict  X_T608, __global const float* restrict  X_T577, __global const float* restrict  X_T604, __global const float* restrict  X_I_222, __global const float* restrict  X_I_221)
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
      float LX_T577 = X_T577[gout_idx];
      float LX_T604 = X_T604[gout_idx];
      float LX_I_222 = X_I_222[(i4_gid + i4_tid)];
      float LX_I_221 = X_I_221[(i4_gid + i4_tid)];
      float LX_T605 = (LX_T577 + LX_T604);
      float LX_T607 = (LX_T605 - LX_I_222);
      float LX_T608 = (LX_T607 * LX_I_221);
      X_T605[gout_idx] = LX_T605;
      X_T608[gout_idx] = LX_T608;
    }
  }
}
