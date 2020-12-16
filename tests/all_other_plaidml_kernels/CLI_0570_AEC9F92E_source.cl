#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 53 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1696 }
// Out stride: { 332416, 23744, 1696, 1 }
// Elementwise input X_T1681 shape: fp32(1, 14, 14, 1696):(332416, 23744, 1696, 1):1298.5 KiB
// Elementwise input X_T1704 shape: fp32(1, 14, 14, 1696):(332416, 23744, 1696, 1):1298.5 KiB
// Elementwise input X_I_662 shape: fp32(1696):(1):6.625 KiB
// Elementwise input X_I_661 shape: fp32(1696):(1):6.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1705 = add(X_T1681, X_T1704)
// Elementwise op: [[pid(Sub)]] X_T1707 = sub(X_T1705, X_I_662)
// Elementwise op: [[pid(Mul)]] X_T1708 = mul(X_T1707, X_I_661)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1696):(332416, 23744, 1696, 1):1298.5 KiB
// Computed true ops: 997248
// Computed work groups: 371
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 53, 1
__kernel void kernel_c124_sdk_584(__global float* restrict  X_T1705, __global float* restrict  X_T1708, __global const float* restrict  X_T1681, __global const float* restrict  X_T1704, __global const float* restrict  X_I_662, __global const float* restrict  X_I_661)
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
      int gout_idx = (((23744 * (i2_gid + i2_tid)) + (1696 * i3)) + (i4_gid + i4_tid));
      float LX_T1681 = X_T1681[gout_idx];
      float LX_T1704 = X_T1704[gout_idx];
      float LX_I_662 = X_I_662[(i4_gid + i4_tid)];
      float LX_I_661 = X_I_661[(i4_gid + i4_tid)];
      float LX_T1705 = (LX_T1681 + LX_T1704);
      float LX_T1707 = (LX_T1705 - LX_I_662);
      float LX_T1708 = (LX_T1707 * LX_I_661);
      X_T1705[gout_idx] = LX_T1705;
      X_T1708[gout_idx] = LX_T1708;
    }
  }
}
