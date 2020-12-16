#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1696 }
// Out stride: { 83104, 11872, 1696, 1 }
// Elementwise input X_T2401 shape: fp32(1, 7, 7, 1696):(83104, 11872, 1696, 1):324.625 KiB
// Elementwise input X_T2424 shape: fp32(1, 7, 7, 1696):(83104, 11872, 1696, 1):324.625 KiB
// Elementwise input X_I_943 shape: fp32(1696):(1):6.625 KiB
// Elementwise input X_I_942 shape: fp32(1696):(1):6.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2425 = add(X_T2401, X_T2424)
// Elementwise op: [[pid(Sub)]] X_T2427 = sub(X_T2425, X_I_943)
// Elementwise op: [[pid(Mul)]] X_T2428 = mul(X_T2427, X_I_942)
// Tile size: { 1, 1, 1, 1696 }
// Contraction output var shape: fp32(1, 7, 7, 1696):(83104, 11872, 1696, 1):324.625 KiB
// Computed true ops: 249312
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 848
// Computed mem write: 13568
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_842(__global float* restrict  X_T2425, __global float* restrict  X_T2428, __global const float* restrict  X_T2401, __global const float* restrict  X_T2424, __global const float* restrict  X_I_943, __global const float* restrict  X_I_942)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 160));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11872 * i2_gid) + (1696 * i3_gid)) + i4);
      float LX_T2401 = X_T2401[gout_idx];
      float LX_T2424 = X_T2424[gout_idx];
      float LX_I_943 = X_I_943[i4];
      float LX_I_942 = X_I_942[i4];
      float LX_T2425 = (LX_T2401 + LX_T2424);
      float LX_T2427 = (LX_T2425 - LX_I_943);
      float LX_T2428 = (LX_T2427 * LX_I_942);
      X_T2425[gout_idx] = LX_T2425;
      X_T2428[gout_idx] = LX_T2428;
    }
  }
}
