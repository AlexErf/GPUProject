#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 800 }
// Out stride: { 39200, 5600, 800, 1 }
// Elementwise input X_T1493 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_T1516 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_I_583 shape: fp32(800):(1):3.125 KiB
// Elementwise input X_I_582 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1517 = add(X_T1493, X_T1516)
// Elementwise op: [[pid(Sub)]] X_T1519 = sub(X_T1517, X_I_583)
// Elementwise op: [[pid(Mul)]] X_T1520 = mul(X_T1519, X_I_582)
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
__kernel void kernel_c108_sdk_518(__global float* restrict  X_T1517, __global float* restrict  X_T1520, __global const float* restrict  X_T1493, __global const float* restrict  X_T1516, __global const float* restrict  X_I_583, __global const float* restrict  X_I_582)
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
      float LX_T1493 = X_T1493[gout_idx];
      float LX_T1516 = X_T1516[gout_idx];
      float LX_I_583 = X_I_583[i4];
      float LX_I_582 = X_I_582[i4];
      float LX_T1517 = (LX_T1493 + LX_T1516);
      float LX_T1519 = (LX_T1517 - LX_I_583);
      float LX_T1520 = (LX_T1519 * LX_I_582);
      X_T1517[gout_idx] = LX_T1517;
      X_T1520[gout_idx] = LX_T1520;
    }
  }
}
