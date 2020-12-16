#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 320 }
// Out stride: { 250880, 8960, 320, 1 }
// Elementwise input X_T413 shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Elementwise input X_T417 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_149 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T418 = div(X_T413, X_T417)
// Elementwise op: [[pid(Add, Switch)]] X_T419 = add(X_T418, X_I_149)
// Elementwise op: X_T420 = cmp_lt(X_T419, X_T2)
// Elementwise op: [[pid(Relu)]] X_T421 = cond(X_T420, X_T2, X_T419)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Computed true ops: 1003520
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2560, 14, 1
__kernel void kernel_c124_sdk_122(__global float* restrict  X_T421, __global const float* restrict  X_T413, __global const float* restrict  X_T417, __global const float* restrict  X_I_149)
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
    float LX_T413 = X_T413[gout_idx];
    float LX_T417 = X_T417[(i4_gid + i4_tid)];
    float LX_I_149 = X_I_149[(i4_gid + i4_tid)];
    float LX_T418 = (LX_T413 / LX_T417);
    float LX_T419 = (LX_T418 + LX_I_149);
    int LX_T420 = (LX_T419 < 0.0f);
    float LX_T421 = select((float)LX_T419, (float)0.0f, (int)LX_T420);
    X_T421[gout_idx] = LX_T421;
  }
}
