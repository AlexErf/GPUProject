#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1280 }
// Out stride: { 62720, 8960, 1280, 1 }
// Elementwise input X_T707 shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Elementwise input X_T711 shape: fp32(1280):(1):5 KiB
// Elementwise input X_I_2 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T712 = div(X_T707, X_T711)
// Elementwise op: [[pid(Add, Switch)]] X_T713 = add(X_T712, X_I_2)
// Elementwise op: X_T714 = cmp_lt(X_T713, X_T3)
// Elementwise op: [[pid(Relu)]] X_T715 = cond(X_T714, X_T3, X_T713)
// Elementwise op: X_T716 = cmp_lt(X_T715, X_T2)
// Elementwise op: [[pid(Relu)]] X_T717 = cond(X_T716, X_T715, X_T2)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Computed true ops: 376320
// Computed work groups: 70
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 10, 1
__kernel void kernel_c43_sdk_194(__global float* restrict  X_T717, __global const float* restrict  X_T707, __global const float* restrict  X_T711, __global const float* restrict  X_I_2)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((8960 * i2_gid) + (1280 * i3_tid)) + (i4_gid + i4));
      float LX_T707 = X_T707[gout_idx];
      float LX_T711 = X_T711[(i4_gid + i4)];
      float LX_I_2 = X_I_2[(i4_gid + i4)];
      float LX_T712 = (LX_T707 / LX_T711);
      float LX_T713 = (LX_T712 + LX_I_2);
      int LX_T714 = (LX_T713 < 0.0f);
      float LX_T715 = select((float)LX_T713, (float)0.0f, (int)LX_T714);
      int LX_T716 = (LX_T715 < 6.0f);
      float LX_T717 = select((float)6.0f, (float)LX_T715, (int)LX_T716);
      X_T717[gout_idx] = LX_T717;
    }
  }
}
