#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 45 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1440 }
// Out stride: { 282240, 20160, 1440, 1 }
// Elementwise input X_T1481 shape: fp32(1, 14, 14, 1440):(282240, 20160, 1440, 1):1102.5 KiB
// Elementwise input X_T1504 shape: fp32(1, 14, 14, 1440):(282240, 20160, 1440, 1):1102.5 KiB
// Elementwise input X_I_582 shape: fp32(1440):(1):5.625 KiB
// Elementwise input X_I_581 shape: fp32(1440):(1):5.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1505 = add(X_T1481, X_T1504)
// Elementwise op: [[pid(Sub)]] X_T1507 = sub(X_T1505, X_I_582)
// Elementwise op: [[pid(Mul)]] X_T1508 = mul(X_T1507, X_I_581)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1440):(282240, 20160, 1440, 1):1102.5 KiB
// Computed true ops: 846720
// Computed work groups: 315
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 45, 1
__kernel void kernel_c124_sdk_512(__global float* restrict  X_T1505, __global float* restrict  X_T1508, __global const float* restrict  X_T1481, __global const float* restrict  X_T1504, __global const float* restrict  X_I_582, __global const float* restrict  X_I_581)
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
      int gout_idx = (((20160 * (i2_gid + i2_tid)) + (1440 * i3)) + (i4_gid + i4_tid));
      float LX_T1481 = X_T1481[gout_idx];
      float LX_T1504 = X_T1504[gout_idx];
      float LX_I_582 = X_I_582[(i4_gid + i4_tid)];
      float LX_I_581 = X_I_581[(i4_gid + i4_tid)];
      float LX_T1505 = (LX_T1481 + LX_T1504);
      float LX_T1507 = (LX_T1505 - LX_I_582);
      float LX_T1508 = (LX_T1507 * LX_I_581);
      X_T1505[gout_idx] = LX_T1505;
      X_T1508[gout_idx] = LX_T1508;
    }
  }
}
