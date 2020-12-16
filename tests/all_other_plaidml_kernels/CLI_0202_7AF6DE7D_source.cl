#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 88 }
// Out stride: { 17248, 1232, 88, 1 }
// Elementwise input X_T1386 shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Elementwise input X_T1390 shape: fp32(88):(1):352 bytes
// Elementwise input X_I_503 shape: fp32(88):(1):352 bytes
// Elementwise input X_T1212 shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1391 = div(X_T1386, X_T1390)
// Elementwise op: [[pid(Add, Switch)]] X_T1392 = add(X_T1391, X_I_503)
// Elementwise op: [[pid(Add)]] X_T1393 = add(X_T1392, X_T1212)
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
__kernel void kernel_c42_sdk_523(__global float* restrict  X_T1393, __global const float* restrict  X_T1386, __global const float* restrict  X_T1390, __global const float* restrict  X_I_503, __global const float* restrict  X_T1212)
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
      float LX_T1386 = X_T1386[gout_idx];
      float LX_T1390 = X_T1390[i4];
      float LX_I_503 = X_I_503[i4];
      float LX_T1212 = X_T1212[gout_idx];
      float LX_T1391 = (LX_T1386 / LX_T1390);
      float LX_T1392 = (LX_T1391 + LX_I_503);
      float LX_T1393 = (LX_T1392 + LX_T1212);
      X_T1393[gout_idx] = LX_T1393;
    }
  }
}
