#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 54 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1728 }
// Out stride: { 338688, 24192, 1728, 1 }
// Elementwise input X_T1733 shape: fp32(1, 14, 14, 1728):(338688, 24192, 1728, 1):1323 KiB
// Elementwise input X_T1737 shape: fp32(1728):(1):6.75 KiB
// Elementwise input X_I_670 shape: fp32(1728):(1):6.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1738 = div(X_T1733, X_T1737)
// Elementwise op: [[pid(Add, Switch)]] X_T1739 = add(X_T1738, X_I_670)
// Elementwise op: X_T1740 = cmp_lt(X_T1739, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1741 = cond(X_T1740, X_T2, X_T1739)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1728):(338688, 24192, 1728, 1):1323 KiB
// Computed true ops: 1354752
// Computed work groups: 378
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 54, 1
__kernel void kernel_c124_sdk_596(__global float* restrict  X_T1741, __global const float* restrict  X_T1733, __global const float* restrict  X_T1737, __global const float* restrict  X_I_670)
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
      int gout_idx = (((24192 * (i2_gid + i2_tid)) + (1728 * i3)) + (i4_gid + i4_tid));
      float LX_T1733 = X_T1733[gout_idx];
      float LX_T1737 = X_T1737[(i4_gid + i4_tid)];
      float LX_I_670 = X_I_670[(i4_gid + i4_tid)];
      float LX_T1738 = (LX_T1733 / LX_T1737);
      float LX_T1739 = (LX_T1738 + LX_I_670);
      int LX_T1740 = (LX_T1739 < 0.0f);
      float LX_T1741 = select((float)LX_T1739, (float)0.0f, (int)LX_T1740);
      X_T1741[gout_idx] = LX_T1741;
    }
  }
}
