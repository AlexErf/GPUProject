#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 54 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1728 }
// Out stride: { 338688, 24192, 1728, 1 }
// Elementwise input X_T1706 shape: fp32(1, 14, 14, 1728):(338688, 24192, 1728, 1):1323 KiB
// Elementwise input X_T1729 shape: fp32(1, 14, 14, 1728):(338688, 24192, 1728, 1):1323 KiB
// Elementwise input X_I_672 shape: fp32(1728):(1):6.75 KiB
// Elementwise input X_I_671 shape: fp32(1728):(1):6.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1730 = add(X_T1706, X_T1729)
// Elementwise op: [[pid(Sub)]] X_T1732 = sub(X_T1730, X_I_672)
// Elementwise op: [[pid(Mul)]] X_T1733 = mul(X_T1732, X_I_671)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1728):(338688, 24192, 1728, 1):1323 KiB
// Computed true ops: 1016064
// Computed work groups: 378
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 54, 1
__kernel void kernel_c124_sdk_593(__global float* restrict  X_T1730, __global float* restrict  X_T1733, __global const float* restrict  X_T1706, __global const float* restrict  X_T1729, __global const float* restrict  X_I_672, __global const float* restrict  X_I_671)
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
      float LX_T1706 = X_T1706[gout_idx];
      float LX_T1729 = X_T1729[gout_idx];
      float LX_I_672 = X_I_672[(i4_gid + i4_tid)];
      float LX_I_671 = X_I_671[(i4_gid + i4_tid)];
      float LX_T1730 = (LX_T1706 + LX_T1729);
      float LX_T1732 = (LX_T1730 - LX_I_672);
      float LX_T1733 = (LX_T1732 * LX_I_671);
      X_T1730[gout_idx] = LX_T1730;
      X_T1733[gout_idx] = LX_T1733;
    }
  }
}
