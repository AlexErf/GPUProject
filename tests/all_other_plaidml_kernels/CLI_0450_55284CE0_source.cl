#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 33 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1056 }
// Out stride: { 206976, 14784, 1056, 1 }
// Elementwise input X_T1181 shape: fp32(1, 14, 14, 1056):(206976, 14784, 1056, 1):808.5 KiB
// Elementwise input X_T1204 shape: fp32(1, 14, 14, 1056):(206976, 14784, 1056, 1):808.5 KiB
// Elementwise input X_I_462 shape: fp32(1056):(1):4.125 KiB
// Elementwise input X_I_461 shape: fp32(1056):(1):4.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1205 = add(X_T1181, X_T1204)
// Elementwise op: [[pid(Sub)]] X_T1207 = sub(X_T1205, X_I_462)
// Elementwise op: [[pid(Mul)]] X_T1208 = mul(X_T1207, X_I_461)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1056):(206976, 14784, 1056, 1):808.5 KiB
// Computed true ops: 620928
// Computed work groups: 231
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 33, 1
__kernel void kernel_c124_sdk_404(__global float* restrict  X_T1205, __global float* restrict  X_T1208, __global const float* restrict  X_T1181, __global const float* restrict  X_T1204, __global const float* restrict  X_I_462, __global const float* restrict  X_I_461)
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
      int gout_idx = (((14784 * (i2_gid + i2_tid)) + (1056 * i3)) + (i4_gid + i4_tid));
      float LX_T1181 = X_T1181[gout_idx];
      float LX_T1204 = X_T1204[gout_idx];
      float LX_I_462 = X_I_462[(i4_gid + i4_tid)];
      float LX_I_461 = X_I_461[(i4_gid + i4_tid)];
      float LX_T1205 = (LX_T1181 + LX_T1204);
      float LX_T1207 = (LX_T1205 - LX_I_462);
      float LX_T1208 = (LX_T1207 * LX_I_461);
      X_T1205[gout_idx] = LX_T1205;
      X_T1208[gout_idx] = LX_T1208;
    }
  }
}
