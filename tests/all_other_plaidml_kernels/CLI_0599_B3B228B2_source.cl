#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1408 }
// Out stride: { 68992, 9856, 1408, 1 }
// Elementwise input X_T1968 shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Elementwise input X_T1991 shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Elementwise input X_I_773 shape: fp32(1408):(1):5.5 KiB
// Elementwise input X_I_772 shape: fp32(1408):(1):5.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1992 = add(X_T1968, X_T1991)
// Elementwise op: [[pid(Sub)]] X_T1994 = sub(X_T1992, X_I_773)
// Elementwise op: [[pid(Mul)]] X_T1995 = mul(X_T1994, X_I_772)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Computed true ops: 206976
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c108_sdk_689(__global float* restrict  X_T1992, __global float* restrict  X_T1995, __global const float* restrict  X_T1968, __global const float* restrict  X_T1991, __global const float* restrict  X_I_773, __global const float* restrict  X_I_772)
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
      int gout_idx = (((9856 * i2_gid) + (1408 * i3_tid)) + (i4_gid + i4));
      float LX_T1968 = X_T1968[gout_idx];
      float LX_T1991 = X_T1991[gout_idx];
      float LX_I_773 = X_I_773[(i4_gid + i4)];
      float LX_I_772 = X_I_772[(i4_gid + i4)];
      float LX_T1992 = (LX_T1968 + LX_T1991);
      float LX_T1994 = (LX_T1992 - LX_I_773);
      float LX_T1995 = (LX_T1994 * LX_I_772);
      X_T1992[gout_idx] = LX_T1992;
      X_T1995[gout_idx] = LX_T1995;
    }
  }
}
