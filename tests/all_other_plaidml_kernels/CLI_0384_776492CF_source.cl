#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 26 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 832 }
// Out stride: { 163072, 11648, 832, 1 }
// Elementwise input X_T998 shape: fp32(1, 14, 14, 832):(163072, 11648, 832, 1):637 KiB
// Elementwise input X_T1021 shape: fp32(1, 14, 14, 832):(163072, 11648, 832, 1):637 KiB
// Elementwise input X_I_392 shape: fp32(832):(1):3.25 KiB
// Elementwise input X_I_391 shape: fp32(832):(1):3.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1022 = add(X_T998, X_T1021)
// Elementwise op: [[pid(Sub)]] X_T1024 = sub(X_T1022, X_I_392)
// Elementwise op: [[pid(Mul)]] X_T1025 = mul(X_T1024, X_I_391)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 832):(163072, 11648, 832, 1):637 KiB
// Computed true ops: 489216
// Computed work groups: 182
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 26, 1
__kernel void kernel_c108_sdk_341(__global float* restrict  X_T1022, __global float* restrict  X_T1025, __global const float* restrict  X_T998, __global const float* restrict  X_T1021, __global const float* restrict  X_I_392, __global const float* restrict  X_I_391)
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
      int gout_idx = (((11648 * (i2_gid + i2_tid)) + (832 * i3)) + (i4_gid + i4_tid));
      float LX_T998 = X_T998[gout_idx];
      float LX_T1021 = X_T1021[gout_idx];
      float LX_I_392 = X_I_392[(i4_gid + i4_tid)];
      float LX_I_391 = X_I_391[(i4_gid + i4_tid)];
      float LX_T1022 = (LX_T998 + LX_T1021);
      float LX_T1024 = (LX_T1022 - LX_I_392);
      float LX_T1025 = (LX_T1024 * LX_I_391);
      X_T1022[gout_idx] = LX_T1022;
      X_T1025[gout_idx] = LX_T1025;
    }
  }
}
