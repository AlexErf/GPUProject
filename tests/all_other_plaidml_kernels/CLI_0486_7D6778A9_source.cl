#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 39 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1248 }
// Out stride: { 244608, 17472, 1248, 1 }
// Elementwise input X_T1331 shape: fp32(1, 14, 14, 1248):(244608, 17472, 1248, 1):955.5 KiB
// Elementwise input X_T1354 shape: fp32(1, 14, 14, 1248):(244608, 17472, 1248, 1):955.5 KiB
// Elementwise input X_I_522 shape: fp32(1248):(1):4.875 KiB
// Elementwise input X_I_521 shape: fp32(1248):(1):4.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1355 = add(X_T1331, X_T1354)
// Elementwise op: [[pid(Sub)]] X_T1357 = sub(X_T1355, X_I_522)
// Elementwise op: [[pid(Mul)]] X_T1358 = mul(X_T1357, X_I_521)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1248):(244608, 17472, 1248, 1):955.5 KiB
// Computed true ops: 733824
// Computed work groups: 273
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 39, 1
__kernel void kernel_c124_sdk_458(__global float* restrict  X_T1355, __global float* restrict  X_T1358, __global const float* restrict  X_T1331, __global const float* restrict  X_T1354, __global const float* restrict  X_I_522, __global const float* restrict  X_I_521)
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
      int gout_idx = (((17472 * (i2_gid + i2_tid)) + (1248 * i3)) + (i4_gid + i4_tid));
      float LX_T1331 = X_T1331[gout_idx];
      float LX_T1354 = X_T1354[gout_idx];
      float LX_I_522 = X_I_522[(i4_gid + i4_tid)];
      float LX_I_521 = X_I_521[(i4_gid + i4_tid)];
      float LX_T1355 = (LX_T1331 + LX_T1354);
      float LX_T1357 = (LX_T1355 - LX_I_522);
      float LX_T1358 = (LX_T1357 * LX_I_521);
      X_T1355[gout_idx] = LX_T1355;
      X_T1358[gout_idx] = LX_T1358;
    }
  }
}
