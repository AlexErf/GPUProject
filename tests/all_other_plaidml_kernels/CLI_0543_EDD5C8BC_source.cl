#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 48 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1536 }
// Out stride: { 301056, 21504, 1536, 1 }
// Elementwise input X_T1583 shape: fp32(1, 14, 14, 1536):(301056, 21504, 1536, 1):1176 KiB
// Elementwise input X_T1587 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_610 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1588 = div(X_T1583, X_T1587)
// Elementwise op: [[pid(Add, Switch)]] X_T1589 = add(X_T1588, X_I_610)
// Elementwise op: X_T1590 = cmp_lt(X_T1589, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1591 = cond(X_T1590, X_T2, X_T1589)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1536):(301056, 21504, 1536, 1):1176 KiB
// Computed true ops: 1204224
// Computed work groups: 336
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 48, 1
__kernel void kernel_c124_sdk_542(__global float* restrict  X_T1591, __global const float* restrict  X_T1583, __global const float* restrict  X_T1587, __global const float* restrict  X_I_610)
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
      int gout_idx = (((21504 * (i2_gid + i2_tid)) + (1536 * i3)) + (i4_gid + i4_tid));
      float LX_T1583 = X_T1583[gout_idx];
      float LX_T1587 = X_T1587[(i4_gid + i4_tid)];
      float LX_I_610 = X_I_610[(i4_gid + i4_tid)];
      float LX_T1588 = (LX_T1583 / LX_T1587);
      float LX_T1589 = (LX_T1588 + LX_I_610);
      int LX_T1590 = (LX_T1589 < 0.0f);
      float LX_T1591 = select((float)LX_T1589, (float)0.0f, (int)LX_T1590);
      X_T1591[gout_idx] = LX_T1591;
    }
  }
}