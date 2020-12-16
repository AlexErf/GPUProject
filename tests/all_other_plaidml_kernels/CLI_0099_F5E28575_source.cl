#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 144 }
// Out stride: { 112896, 4032, 144, 1 }
// Elementwise input X_T191 shape: fp32(1, 28, 28, 144):(112896, 4032, 144, 1):441 KiB
// Elementwise input X_T195 shape: fp32(144):(1):576 bytes
// Elementwise input X_I_58 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T196 = div(X_T191, X_T195)
// Elementwise op: [[pid(Add, Switch)]] X_T197 = add(X_T196, X_I_58)
// Elementwise op: X_T198 = cmp_lt(X_T197, X_T3)
// Elementwise op: [[pid(Relu)]] X_T199 = cond(X_T198, X_T3, X_T197)
// Elementwise op: X_T200 = cmp_lt(X_T199, X_T2)
// Elementwise op: [[pid(Relu)]] X_T201 = cond(X_T200, X_T199, X_T2)
// Tile size: { 1, 4, 1, 144 }
// Contraction output var shape: fp32(1, 28, 28, 144):(112896, 4032, 144, 1):441 KiB
// Computed true ops: 677376
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
__kernel void kernel_c43_sdk_47(__global float* restrict  X_T201, __global const float* restrict  X_T191, __global const float* restrict  X_T195, __global const float* restrict  X_I_58)
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
      float LX_T191 = X_T191[gout_idx];
      float LX_T195 = X_T195[i4];
      float LX_I_58 = X_I_58[i4];
      float LX_T196 = (LX_T191 / LX_T195);
      float LX_T197 = (LX_T196 + LX_I_58);
      int LX_T198 = (LX_T197 < 0.0f);
      float LX_T199 = select((float)LX_T197, (float)0.0f, (int)LX_T198);
      int LX_T200 = (LX_T199 < 6.0f);
      float LX_T201 = select((float)6.0f, (float)LX_T199, (int)LX_T200);
      X_T201[gout_idx] = LX_T201;
    }
  }
}
