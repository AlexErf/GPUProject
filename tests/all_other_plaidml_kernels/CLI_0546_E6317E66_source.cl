#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 49 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1568 }
// Out stride: { 307328, 21952, 1568, 1 }
// Elementwise input X_T1581 shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Elementwise input X_T1604 shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Elementwise input X_I_622 shape: fp32(1568):(1):6.125 KiB
// Elementwise input X_I_621 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1605 = add(X_T1581, X_T1604)
// Elementwise op: [[pid(Sub)]] X_T1607 = sub(X_T1605, X_I_622)
// Elementwise op: [[pid(Mul)]] X_T1608 = mul(X_T1607, X_I_621)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Computed true ops: 921984
// Computed work groups: 343
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 49, 1
__kernel void kernel_c124_sdk_548(__global float* restrict  X_T1605, __global float* restrict  X_T1608, __global const float* restrict  X_T1581, __global const float* restrict  X_T1604, __global const float* restrict  X_I_622, __global const float* restrict  X_I_621)
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
      int gout_idx = (((21952 * (i2_gid + i2_tid)) + (1568 * i3)) + (i4_gid + i4_tid));
      float LX_T1581 = X_T1581[gout_idx];
      float LX_T1604 = X_T1604[gout_idx];
      float LX_I_622 = X_I_622[(i4_gid + i4_tid)];
      float LX_I_621 = X_I_621[(i4_gid + i4_tid)];
      float LX_T1605 = (LX_T1581 + LX_T1604);
      float LX_T1607 = (LX_T1605 - LX_I_622);
      float LX_T1608 = (LX_T1607 * LX_I_621);
      X_T1605[gout_idx] = LX_T1605;
      X_T1608[gout_idx] = LX_T1608;
    }
  }
}
