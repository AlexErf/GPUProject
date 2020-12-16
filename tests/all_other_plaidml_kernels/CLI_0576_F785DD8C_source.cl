#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1280 }
// Out stride: { 62720, 8960, 1280, 1 }
// Elementwise input X_T1868 shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Elementwise input X_T1891 shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Elementwise input X_I_733 shape: fp32(1280):(1):5 KiB
// Elementwise input X_I_732 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1892 = add(X_T1868, X_T1891)
// Elementwise op: [[pid(Sub)]] X_T1894 = sub(X_T1892, X_I_733)
// Elementwise op: [[pid(Mul)]] X_T1895 = mul(X_T1894, X_I_732)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Computed true ops: 188160
// Computed work groups: 70
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 10, 1
__kernel void kernel_c108_sdk_653(__global float* restrict  X_T1892, __global float* restrict  X_T1895, __global const float* restrict  X_T1868, __global const float* restrict  X_T1891, __global const float* restrict  X_I_733, __global const float* restrict  X_I_732)
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
      int gout_idx = (((8960 * i2_gid) + (1280 * i3_tid)) + (i4_gid + i4));
      float LX_T1868 = X_T1868[gout_idx];
      float LX_T1891 = X_T1891[gout_idx];
      float LX_I_733 = X_I_733[(i4_gid + i4)];
      float LX_I_732 = X_I_732[(i4_gid + i4)];
      float LX_T1892 = (LX_T1868 + LX_T1891);
      float LX_T1894 = (LX_T1892 - LX_I_733);
      float LX_T1895 = (LX_T1894 * LX_I_732);
      X_T1892[gout_idx] = LX_T1892;
      X_T1895[gout_idx] = LX_T1895;
    }
  }
}
