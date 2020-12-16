#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1408 }
// Out stride: { 68992, 9856, 1408, 1 }
// Elementwise input X_T2176 shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Elementwise input X_T2199 shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Elementwise input X_I_853 shape: fp32(1408):(1):5.5 KiB
// Elementwise input X_I_852 shape: fp32(1408):(1):5.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2200 = add(X_T2176, X_T2199)
// Elementwise op: [[pid(Sub)]] X_T2202 = sub(X_T2200, X_I_853)
// Elementwise op: [[pid(Mul)]] X_T2203 = mul(X_T2202, X_I_852)
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
__kernel void kernel_c124_sdk_761(__global float* restrict  X_T2200, __global float* restrict  X_T2203, __global const float* restrict  X_T2176, __global const float* restrict  X_T2199, __global const float* restrict  X_I_853, __global const float* restrict  X_I_852)
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
      float LX_T2176 = X_T2176[gout_idx];
      float LX_T2199 = X_T2199[gout_idx];
      float LX_I_853 = X_I_853[(i4_gid + i4)];
      float LX_I_852 = X_I_852[(i4_gid + i4)];
      float LX_T2200 = (LX_T2176 + LX_T2199);
      float LX_T2202 = (LX_T2200 - LX_I_853);
      float LX_T2203 = (LX_T2202 * LX_I_852);
      X_T2200[gout_idx] = LX_T2200;
      X_T2203[gout_idx] = LX_T2203;
    }
  }
}
