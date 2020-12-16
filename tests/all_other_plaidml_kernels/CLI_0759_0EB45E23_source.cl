#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1920 }
// Out stride: { 94080, 13440, 1920, 1 }
// Elementwise input X_T2576 shape: fp32(1, 7, 7, 1920):(94080, 13440, 1920, 1):367.5 KiB
// Elementwise input X_T2599 shape: fp32(1, 7, 7, 1920):(94080, 13440, 1920, 1):367.5 KiB
// Elementwise input X_I_4 shape: fp32(1920):(1):7.5 KiB
// Elementwise input X_I_3 shape: fp32(1920):(1):7.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2600 = add(X_T2576, X_T2599)
// Elementwise op: [[pid(Sub)]] X_T2601 = sub(X_T2600, X_I_4)
// Elementwise op: [[pid(Mul)]] X_T2602 = mul(X_T2601, X_I_3)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1920):(94080, 13440, 1920, 1):367.5 KiB
// Computed true ops: 282240
// Computed work groups: 105
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 15, 1
__kernel void kernel_c124_sdk_905(__global float* restrict  X_T2602, __global const float* restrict  X_T2576, __global const float* restrict  X_T2599, __global const float* restrict  X_I_4, __global const float* restrict  X_I_3)
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
      int gout_idx = (((13440 * i2_gid) + (1920 * i3_tid)) + (i4_gid + i4));
      float LX_T2576 = X_T2576[gout_idx];
      float LX_T2599 = X_T2599[gout_idx];
      float LX_I_4 = X_I_4[(i4_gid + i4)];
      float LX_I_3 = X_I_3[(i4_gid + i4)];
      float LX_T2600 = (LX_T2576 + LX_T2599);
      float LX_T2601 = (LX_T2600 - LX_I_4);
      float LX_T2602 = (LX_T2601 * LX_I_3);
      X_T2602[gout_idx] = LX_T2602;
    }
  }
}
