#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 41 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1312 }
// Out stride: { 257152, 18368, 1312, 1 }
// Elementwise input X_T1381 shape: fp32(1, 14, 14, 1312):(257152, 18368, 1312, 1):1004.5 KiB
// Elementwise input X_T1404 shape: fp32(1, 14, 14, 1312):(257152, 18368, 1312, 1):1004.5 KiB
// Elementwise input X_I_542 shape: fp32(1312):(1):5.125 KiB
// Elementwise input X_I_541 shape: fp32(1312):(1):5.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1405 = add(X_T1381, X_T1404)
// Elementwise op: [[pid(Sub)]] X_T1407 = sub(X_T1405, X_I_542)
// Elementwise op: [[pid(Mul)]] X_T1408 = mul(X_T1407, X_I_541)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1312):(257152, 18368, 1312, 1):1004.5 KiB
// Computed true ops: 771456
// Computed work groups: 287
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 41, 1
__kernel void kernel_c124_sdk_476(__global float* restrict  X_T1405, __global float* restrict  X_T1408, __global const float* restrict  X_T1381, __global const float* restrict  X_T1404, __global const float* restrict  X_I_542, __global const float* restrict  X_I_541)
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
      int gout_idx = (((18368 * (i2_gid + i2_tid)) + (1312 * i3)) + (i4_gid + i4_tid));
      float LX_T1381 = X_T1381[gout_idx];
      float LX_T1404 = X_T1404[gout_idx];
      float LX_I_542 = X_I_542[(i4_gid + i4_tid)];
      float LX_I_541 = X_I_541[(i4_gid + i4_tid)];
      float LX_T1405 = (LX_T1381 + LX_T1404);
      float LX_T1407 = (LX_T1405 - LX_I_542);
      float LX_T1408 = (LX_T1407 * LX_I_541);
      X_T1405[gout_idx] = LX_T1405;
      X_T1408[gout_idx] = LX_T1408;
    }
  }
}
