#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 320 }
// Out stride: { 250880, 8960, 320, 1 }
// Elementwise input X_T386 shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Elementwise input X_T409 shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Elementwise input X_I_151 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_150 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T410 = add(X_T386, X_T409)
// Elementwise op: [[pid(Sub)]] X_T412 = sub(X_T410, X_I_151)
// Elementwise op: [[pid(Mul)]] X_T413 = mul(X_T412, X_I_150)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Computed true ops: 752640
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 14, 1
__kernel void kernel_c124_sdk_119(__global float* restrict  X_T410, __global float* restrict  X_T413, __global const float* restrict  X_T386, __global const float* restrict  X_T409, __global const float* restrict  X_I_151, __global const float* restrict  X_I_150)
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
    int gout_idx = (((8960 * i2) + (320 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T386 = X_T386[gout_idx];
    float LX_T409 = X_T409[gout_idx];
    float LX_I_151 = X_I_151[(i4_gid + i4_tid)];
    float LX_I_150 = X_I_150[(i4_gid + i4_tid)];
    float LX_T410 = (LX_T386 + LX_T409);
    float LX_T412 = (LX_T410 - LX_I_151);
    float LX_T413 = (LX_T412 * LX_I_150);
    X_T410[gout_idx] = LX_T410;
    X_T413[gout_idx] = LX_T413;
  }
}
