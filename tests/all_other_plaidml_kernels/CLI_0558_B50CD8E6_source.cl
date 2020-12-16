#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 51 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1632 }
// Out stride: { 319872, 22848, 1632, 1 }
// Elementwise input X_T1631 shape: fp32(1, 14, 14, 1632):(319872, 22848, 1632, 1):1249.5 KiB
// Elementwise input X_T1654 shape: fp32(1, 14, 14, 1632):(319872, 22848, 1632, 1):1249.5 KiB
// Elementwise input X_I_642 shape: fp32(1632):(1):6.375 KiB
// Elementwise input X_I_641 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1655 = add(X_T1631, X_T1654)
// Elementwise op: [[pid(Sub)]] X_T1657 = sub(X_T1655, X_I_642)
// Elementwise op: [[pid(Mul)]] X_T1658 = mul(X_T1657, X_I_641)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1632):(319872, 22848, 1632, 1):1249.5 KiB
// Computed true ops: 959616
// Computed work groups: 357
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 51, 1
__kernel void kernel_c124_sdk_566(__global float* restrict  X_T1655, __global float* restrict  X_T1658, __global const float* restrict  X_T1631, __global const float* restrict  X_T1654, __global const float* restrict  X_I_642, __global const float* restrict  X_I_641)
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
      int gout_idx = (((22848 * (i2_gid + i2_tid)) + (1632 * i3)) + (i4_gid + i4_tid));
      float LX_T1631 = X_T1631[gout_idx];
      float LX_T1654 = X_T1654[gout_idx];
      float LX_I_642 = X_I_642[(i4_gid + i4_tid)];
      float LX_I_641 = X_I_641[(i4_gid + i4_tid)];
      float LX_T1655 = (LX_T1631 + LX_T1654);
      float LX_T1657 = (LX_T1655 - LX_I_642);
      float LX_T1658 = (LX_T1657 * LX_I_641);
      X_T1655[gout_idx] = LX_T1655;
      X_T1658[gout_idx] = LX_T1658;
    }
  }
}
