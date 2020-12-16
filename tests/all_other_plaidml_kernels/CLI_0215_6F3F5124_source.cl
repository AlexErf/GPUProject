#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 88 }
// Out stride: { 17248, 1232, 88, 1 }
// Elementwise input X_T1422 shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Elementwise input X_T1434 shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Elementwise input X_I_532 shape: fp32(88):(1):352 bytes
// Elementwise input X_I_531 shape: fp32(88):(1):352 bytes
// Elementwise op: [[pid(Concatenate)]] X_T1435 = add(X_T1422, X_T1434)
// Elementwise op: [[pid(Sub)]] X_T1436 = sub(X_T1435, X_I_532)
// Elementwise op: [[pid(Mul)]] X_T1437 = mul(X_T1436, X_I_531)
// Tile size: { 1, 2, 2, 88 }
// Contraction output var shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Computed true ops: 51744
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_544(__global float* restrict  X_T1437, __global const float* restrict  X_T1422, __global const float* restrict  X_T1434, __global const float* restrict  X_I_532, __global const float* restrict  X_I_531)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 24));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((1232 * (i2_gid + i2_tid)) + (88 * (i3_gid + i3_tid))) + i4);
      float LX_T1422 = X_T1422[gout_idx];
      float LX_T1434 = X_T1434[gout_idx];
      float LX_I_532 = X_I_532[i4];
      float LX_I_531 = X_I_531[i4];
      float LX_T1435 = (LX_T1422 + LX_T1434);
      float LX_T1436 = (LX_T1435 - LX_I_532);
      float LX_T1437 = (LX_T1436 * LX_I_531);
      X_T1437[gout_idx] = LX_T1437;
    }
  }
}
