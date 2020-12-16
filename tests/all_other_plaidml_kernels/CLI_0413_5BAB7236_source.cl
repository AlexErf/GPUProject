#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 800 }
// Out stride: { 39200, 5600, 800, 1 }
// Elementwise input X_T1373 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_T1396 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_I_543 shape: fp32(800):(1):3.125 KiB
// Elementwise input X_I_542 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1397 = add(X_T1373, X_T1396)
// Elementwise op: [[pid(Sub)]] X_T1399 = sub(X_T1397, X_I_543)
// Elementwise op: [[pid(Mul)]] X_T1400 = mul(X_T1399, X_I_542)
// Tile size: { 1, 1, 1, 800 }
// Contraction output var shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Computed true ops: 117600
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 400
// Computed mem write: 6400
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_482(__global float* restrict  X_T1397, __global float* restrict  X_T1400, __global const float* restrict  X_T1373, __global const float* restrict  X_T1396, __global const float* restrict  X_I_543, __global const float* restrict  X_I_542)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((5600 * i2_gid) + (800 * i3_gid)) + i4);
      float LX_T1373 = X_T1373[gout_idx];
      float LX_T1396 = X_T1396[gout_idx];
      float LX_I_543 = X_I_543[i4];
      float LX_I_542 = X_I_542[i4];
      float LX_T1397 = (LX_T1373 + LX_T1396);
      float LX_T1399 = (LX_T1397 - LX_I_543);
      float LX_T1400 = (LX_T1399 * LX_I_542);
      X_T1397[gout_idx] = LX_T1397;
      X_T1400[gout_idx] = LX_T1400;
    }
  }
}
