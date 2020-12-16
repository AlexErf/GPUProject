#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1728 }
// Out stride: { 84672, 12096, 1728, 1 }
// Elementwise input X_T2426 shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Elementwise input X_T2449 shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Elementwise input X_I_953 shape: fp32(1728):(1):6.75 KiB
// Elementwise input X_I_952 shape: fp32(1728):(1):6.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2450 = add(X_T2426, X_T2449)
// Elementwise op: [[pid(Sub)]] X_T2452 = sub(X_T2450, X_I_953)
// Elementwise op: [[pid(Mul)]] X_T2453 = mul(X_T2452, X_I_952)
// Tile size: { 1, 1, 1, 1728 }
// Contraction output var shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Computed true ops: 254016
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 864
// Computed mem write: 13824
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_851(__global float* restrict  X_T2450, __global float* restrict  X_T2453, __global const float* restrict  X_T2426, __global const float* restrict  X_T2449, __global const float* restrict  X_I_953, __global const float* restrict  X_I_952)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12096 * i2_gid) + (1728 * i3_gid)) + i4);
      float LX_T2426 = X_T2426[gout_idx];
      float LX_T2449 = X_T2449[gout_idx];
      float LX_I_953 = X_I_953[i4];
      float LX_I_952 = X_I_952[i4];
      float LX_T2450 = (LX_T2426 + LX_T2449);
      float LX_T2452 = (LX_T2450 - LX_I_953);
      float LX_T2453 = (LX_T2452 * LX_I_952);
      X_T2450[gout_idx] = LX_T2450;
      X_T2453[gout_idx] = LX_T2453;
    }
  }
}
