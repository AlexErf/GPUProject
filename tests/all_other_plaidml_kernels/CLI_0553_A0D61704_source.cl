#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1120 }
// Out stride: { 54880, 7840, 1120, 1 }
// Elementwise input X_T1770 shape: fp32(1, 7, 7, 1120):(54880, 7840, 1120, 1):214.375 KiB
// Elementwise input X_T1774 shape: fp32(1120):(1):4.375 KiB
// Elementwise input X_I_681 shape: fp32(1120):(1):4.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1775 = div(X_T1770, X_T1774)
// Elementwise op: [[pid(Add, Switch)]] X_T1776 = add(X_T1775, X_I_681)
// Elementwise op: X_T1777 = cmp_lt(X_T1776, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1778 = cond(X_T1777, X_T2, X_T1776)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1120):(54880, 7840, 1120, 1):214.375 KiB
// Computed true ops: 219520
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c108_sdk_611(__global float* restrict  X_T1778, __global const float* restrict  X_T1770, __global const float* restrict  X_T1774, __global const float* restrict  X_I_681)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 1024));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((7840 * i2_gid) + (1120 * i3_tid)) + (i4_gid + i4));
        float LX_T1770 = X_T1770[gout_idx];
        float LX_T1774 = X_T1774[(i4_gid + i4)];
        float LX_I_681 = X_I_681[(i4_gid + i4)];
        float LX_T1775 = (LX_T1770 / LX_T1774);
        float LX_T1776 = (LX_T1775 + LX_I_681);
        int LX_T1777 = (LX_T1776 < 0.0f);
        float LX_T1778 = select((float)LX_T1776, (float)0.0f, (int)LX_T1777);
        X_T1778[gout_idx] = LX_T1778;
      }
    }
  }
}
