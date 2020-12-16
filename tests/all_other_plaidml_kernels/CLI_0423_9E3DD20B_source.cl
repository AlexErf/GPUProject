#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 864 }
// Out stride: { 42336, 6048, 864, 1 }
// Elementwise input X_T1423 shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Elementwise input X_T1446 shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Elementwise input X_I_563 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_562 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1447 = add(X_T1423, X_T1446)
// Elementwise op: [[pid(Sub)]] X_T1449 = sub(X_T1447, X_I_563)
// Elementwise op: [[pid(Mul)]] X_T1450 = mul(X_T1449, X_I_562)
// Tile size: { 1, 1, 1, 864 }
// Contraction output var shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Computed true ops: 127008
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 432
// Computed mem write: 6912
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_500(__global float* restrict  X_T1447, __global float* restrict  X_T1450, __global const float* restrict  X_T1423, __global const float* restrict  X_T1446, __global const float* restrict  X_I_563, __global const float* restrict  X_I_562)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6048 * i2_gid) + (864 * i3_gid)) + i4);
      float LX_T1423 = X_T1423[gout_idx];
      float LX_T1446 = X_T1446[gout_idx];
      float LX_I_563 = X_I_563[i4];
      float LX_I_562 = X_I_562[i4];
      float LX_T1447 = (LX_T1423 + LX_T1446);
      float LX_T1449 = (LX_T1447 - LX_I_563);
      float LX_T1450 = (LX_T1449 * LX_I_562);
      X_T1447[gout_idx] = LX_T1447;
      X_T1450[gout_idx] = LX_T1450;
    }
  }
}
