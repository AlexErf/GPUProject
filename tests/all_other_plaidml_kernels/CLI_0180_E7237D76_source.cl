#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 336 }
// Out stride: { 592704, 14112, 336, 1 }
// Elementwise input X_T1569 shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Elementwise input X_T1573 shape: fp32(336):(1):1.3125 KiB
// Elementwise input X_I_608 shape: fp32(336):(1):1.3125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1574 = div(X_T1569, X_T1573)
// Elementwise op: [[pid(Add, Switch)]] X_T1575 = add(X_T1574, X_I_608)
// Elementwise op: X_T1576 = cmp_lt(X_T1575, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1577 = cond(X_T1576, X_T1, X_T1575)
// Tile size: { 1, 2, 4, 32 }
// Contraction output var shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Computed true ops: 2370816
// Computed work groups: 2541
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 21
__kernel void kernel_c42_sdk_599(__global float* restrict  X_T1577, __global const float* restrict  X_T1569, __global const float* restrict  X_T1573, __global const float* restrict  X_I_608)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 4);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i4_cond = ((i4_gid != 320) || (i4_tid < 16));
  if (i4_cond)
  {
    int i3_cond = ((i3_gid != 40) || (i3_tid < 2));
    if (i3_cond)
    {
      int gout_idx = (((14112 * (i2_gid + i2_tid)) + (336 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
      float LX_T1569 = X_T1569[gout_idx];
      float LX_T1573 = X_T1573[(i4_gid + i4_tid)];
      float LX_I_608 = X_I_608[(i4_gid + i4_tid)];
      float LX_T1574 = (LX_T1569 / LX_T1573);
      float LX_T1575 = (LX_T1574 + LX_I_608);
      int LX_T1576 = (LX_T1575 < 0.0f);
      float LX_T1577 = select((float)LX_T1575, (float)0.0f, (int)LX_T1576);
      X_T1577[gout_idx] = LX_T1577;
    }
  }
}
