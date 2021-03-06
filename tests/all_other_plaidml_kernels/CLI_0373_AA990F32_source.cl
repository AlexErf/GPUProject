#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 544 }
// Out stride: { 26656, 3808, 544, 1 }
// Elementwise input X_T1169 shape: fp32(1, 7, 7, 544):(26656, 3808, 544, 1):104.125 KiB
// Elementwise input X_T1196 shape: fp32(1, 7, 7, 544):(26656, 3808, 544, 1):104.125 KiB
// Elementwise input X_I_463 shape: fp32(544):(1):2.125 KiB
// Elementwise input X_I_462 shape: fp32(544):(1):2.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1197 = add(X_T1169, X_T1196)
// Elementwise op: [[pid(Sub)]] X_T1199 = sub(X_T1197, X_I_463)
// Elementwise op: [[pid(Mul)]] X_T1200 = mul(X_T1199, X_I_462)
// Tile size: { 1, 1, 1, 544 }
// Contraction output var shape: fp32(1, 7, 7, 544):(26656, 3808, 544, 1):104.125 KiB
// Computed true ops: 79968
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 272
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_410(__global float* restrict  X_T1197, __global float* restrict  X_T1200, __global const float* restrict  X_T1169, __global const float* restrict  X_T1196, __global const float* restrict  X_I_463, __global const float* restrict  X_I_462)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((3808 * i2_gid) + (544 * i3_gid)) + i4);
      float LX_T1169 = X_T1169[gout_idx];
      float LX_T1196 = X_T1196[gout_idx];
      float LX_I_463 = X_I_463[i4];
      float LX_I_462 = X_I_462[i4];
      float LX_T1197 = (LX_T1169 + LX_T1196);
      float LX_T1199 = (LX_T1197 - LX_I_463);
      float LX_T1200 = (LX_T1199 * LX_I_462);
      X_T1197[gout_idx] = LX_T1197;
      X_T1200[gout_idx] = LX_T1200;
    }
  }
}
