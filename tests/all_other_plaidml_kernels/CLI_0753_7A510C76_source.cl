#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1888 }
// Out stride: { 92512, 13216, 1888, 1 }
// Elementwise input X_T2551 shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Elementwise input X_T2574 shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Elementwise input X_I_1003 shape: fp32(1888):(1):7.375 KiB
// Elementwise input X_I_1002 shape: fp32(1888):(1):7.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2575 = add(X_T2551, X_T2574)
// Elementwise op: [[pid(Sub)]] X_T2577 = sub(X_T2575, X_I_1003)
// Elementwise op: [[pid(Mul)]] X_T2578 = mul(X_T2577, X_I_1002)
// Tile size: { 1, 1, 1, 1888 }
// Contraction output var shape: fp32(1, 7, 7, 1888):(92512, 13216, 1888, 1):361.375 KiB
// Computed true ops: 277536
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 944
// Computed mem write: 15104
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_896(__global float* restrict  X_T2575, __global float* restrict  X_T2578, __global const float* restrict  X_T2551, __global const float* restrict  X_T2574, __global const float* restrict  X_I_1003, __global const float* restrict  X_I_1002)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((13216 * i2_gid) + (1888 * i3_gid)) + i4);
      float LX_T2551 = X_T2551[gout_idx];
      float LX_T2574 = X_T2574[gout_idx];
      float LX_I_1003 = X_I_1003[i4];
      float LX_I_1002 = X_I_1002[i4];
      float LX_T2575 = (LX_T2551 + LX_T2574);
      float LX_T2577 = (LX_T2575 - LX_I_1003);
      float LX_T2578 = (LX_T2577 * LX_I_1002);
      X_T2575[gout_idx] = LX_T2575;
      X_T2578[gout_idx] = LX_T2578;
    }
  }
}
