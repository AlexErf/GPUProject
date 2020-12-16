#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 84 }
// Out stride: { 148176, 3528, 84, 1 }
// Elementwise input X_T413 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise input X_T417 shape: fp32(84):(1):336 bytes
// Elementwise input X_I_164 shape: fp32(84):(1):336 bytes
// Elementwise input X_T393 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T418 = div(X_T413, X_T417)
// Elementwise op: [[pid(Add, Switch)]] X_T419 = add(X_T418, X_I_164)
// Elementwise op: [[pid(Add)]] X_T420 = add(X_T393, X_T419)
// Elementwise op: X_T430 = cmp_lt(X_T420, X_T1)
// Elementwise op: [[pid(Relu)]] X_T431 = cond(X_T430, X_T1, X_T420)
// Tile size: { 1, 2, 2, 84 }
// Contraction output var shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Computed true ops: 740880
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_142(__global float* restrict  X_T420, __global float* restrict  X_T431, __global const float* restrict  X_T413, __global const float* restrict  X_T417, __global const float* restrict  X_I_164, __global const float* restrict  X_T393)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 20));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((3528 * (i2_gid + i2_tid)) + (84 * (i3_gid + i3_tid))) + i4);
      float LX_T413 = X_T413[gout_idx];
      float LX_T417 = X_T417[i4];
      float LX_I_164 = X_I_164[i4];
      float LX_T393 = X_T393[gout_idx];
      float LX_T418 = (LX_T413 / LX_T417);
      float LX_T419 = (LX_T418 + LX_I_164);
      float LX_T420 = (LX_T393 + LX_T419);
      int LX_T430 = (LX_T420 < 0.0f);
      float LX_T431 = select((float)LX_T420, (float)0.0f, (int)LX_T430);
      X_T420[gout_idx] = LX_T420;
      X_T431[gout_idx] = LX_T431;
    }
  }
}
