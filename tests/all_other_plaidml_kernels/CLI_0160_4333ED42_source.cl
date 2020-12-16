#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 192 }
// Out stride: { 150528, 5376, 192, 1 }
// Elementwise input X_T258 shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Elementwise input X_T281 shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Elementwise input X_I_111 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_110 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Concatenate)]] X_T282 = add(X_T258, X_T281)
// Elementwise op: [[pid(Sub)]] X_T284 = sub(X_T282, X_I_111)
// Elementwise op: [[pid(Mul)]] X_T285 = mul(X_T284, X_I_110)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Computed true ops: 451584
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c68_sdk_83(__global float* restrict  X_T282, __global float* restrict  X_T285, __global const float* restrict  X_T258, __global const float* restrict  X_T281, __global const float* restrict  X_I_111, __global const float* restrict  X_I_110)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
  {
    int i2 = ((4 * i2_lid) + i2_tid);
    int gout_idx = (((5376 * i2) + (192 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T258 = X_T258[gout_idx];
    float LX_T281 = X_T281[gout_idx];
    float LX_I_111 = X_I_111[(i4_gid + i4_tid)];
    float LX_I_110 = X_I_110[(i4_gid + i4_tid)];
    float LX_T282 = (LX_T258 + LX_T281);
    float LX_T284 = (LX_T282 - LX_I_111);
    float LX_T285 = (LX_T284 * LX_I_110);
    X_T282[gout_idx] = LX_T282;
    X_T285[gout_idx] = LX_T285;
  }
}
