#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 43 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1376 }
// Out stride: { 269696, 19264, 1376, 1 }
// Elementwise input X_T1431 shape: fp32(1, 14, 14, 1376):(269696, 19264, 1376, 1):1053.5 KiB
// Elementwise input X_T1454 shape: fp32(1, 14, 14, 1376):(269696, 19264, 1376, 1):1053.5 KiB
// Elementwise input X_I_562 shape: fp32(1376):(1):5.375 KiB
// Elementwise input X_I_561 shape: fp32(1376):(1):5.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1455 = add(X_T1431, X_T1454)
// Elementwise op: [[pid(Sub)]] X_T1457 = sub(X_T1455, X_I_562)
// Elementwise op: [[pid(Mul)]] X_T1458 = mul(X_T1457, X_I_561)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1376):(269696, 19264, 1376, 1):1053.5 KiB
// Computed true ops: 809088
// Computed work groups: 301
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 43, 1
__kernel void kernel_c124_sdk_494(__global float* restrict  X_T1455, __global float* restrict  X_T1458, __global const float* restrict  X_T1431, __global const float* restrict  X_T1454, __global const float* restrict  X_I_562, __global const float* restrict  X_I_561)
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
      int gout_idx = (((19264 * (i2_gid + i2_tid)) + (1376 * i3)) + (i4_gid + i4_tid));
      float LX_T1431 = X_T1431[gout_idx];
      float LX_T1454 = X_T1454[gout_idx];
      float LX_I_562 = X_I_562[(i4_gid + i4_tid)];
      float LX_I_561 = X_I_561[(i4_gid + i4_tid)];
      float LX_T1455 = (LX_T1431 + LX_T1454);
      float LX_T1457 = (LX_T1455 - LX_I_562);
      float LX_T1458 = (LX_T1457 * LX_I_561);
      X_T1455[gout_idx] = LX_T1455;
      X_T1458[gout_idx] = LX_T1458;
    }
  }
}
