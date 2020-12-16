#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 42 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1344 }
// Out stride: { 263424, 18816, 1344, 1 }
// Elementwise input X_T1406 shape: fp32(1, 14, 14, 1344):(263424, 18816, 1344, 1):1029 KiB
// Elementwise input X_T1429 shape: fp32(1, 14, 14, 1344):(263424, 18816, 1344, 1):1029 KiB
// Elementwise input X_I_552 shape: fp32(1344):(1):5.25 KiB
// Elementwise input X_I_551 shape: fp32(1344):(1):5.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1430 = add(X_T1406, X_T1429)
// Elementwise op: [[pid(Sub)]] X_T1432 = sub(X_T1430, X_I_552)
// Elementwise op: [[pid(Mul)]] X_T1433 = mul(X_T1432, X_I_551)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1344):(263424, 18816, 1344, 1):1029 KiB
// Computed true ops: 790272
// Computed work groups: 294
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 42, 1
__kernel void kernel_c124_sdk_485(__global float* restrict  X_T1430, __global float* restrict  X_T1433, __global const float* restrict  X_T1406, __global const float* restrict  X_T1429, __global const float* restrict  X_I_552, __global const float* restrict  X_I_551)
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
      int gout_idx = (((18816 * (i2_gid + i2_tid)) + (1344 * i3)) + (i4_gid + i4_tid));
      float LX_T1406 = X_T1406[gout_idx];
      float LX_T1429 = X_T1429[gout_idx];
      float LX_I_552 = X_I_552[(i4_gid + i4_tid)];
      float LX_I_551 = X_I_551[(i4_gid + i4_tid)];
      float LX_T1430 = (LX_T1406 + LX_T1429);
      float LX_T1432 = (LX_T1430 - LX_I_552);
      float LX_T1433 = (LX_T1432 * LX_I_551);
      X_T1430[gout_idx] = LX_T1430;
      X_T1433[gout_idx] = LX_T1433;
    }
  }
}
