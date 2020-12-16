#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2560 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 320 }
// Out stride: { 250880, 8960, 320, 1 }
// Elementwise input X_T385 shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Elementwise input X_T389 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_149 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T390 = div(X_T385, X_T389)
// Elementwise op: [[pid(Add, Switch)]] X_T391 = add(X_T390, X_I_149)
// Elementwise op: X_T392 = cmp_lt(X_T391, X_T2)
// Elementwise op: [[pid(Relu)]] X_T393 = cond(X_T392, X_T2, X_T391)
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
__kernel void kernel_c68_sdk_122(__global float* restrict  X_T393, __global const float* restrict  X_T385, __global const float* restrict  X_T389, __global const float* restrict  X_I_149)
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
    float LX_T385 = X_T385[gout_idx];
    float LX_T389 = X_T389[(i4_gid + i4_tid)];
    float LX_I_149 = X_I_149[(i4_gid + i4_tid)];
    float LX_T390 = (LX_T385 / LX_T389);
    float LX_T391 = (LX_T390 + LX_I_149);
    int LX_T392 = (LX_T391 < 0.0f);
    float LX_T393 = select((float)LX_T391, (float)0.0f, (int)LX_T392);
    X_T393[gout_idx] = LX_T393;
  }
}
