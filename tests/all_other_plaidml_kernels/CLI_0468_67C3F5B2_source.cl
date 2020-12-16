#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 36 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1152 }
// Out stride: { 225792, 16128, 1152, 1 }
// Elementwise input X_T1256 shape: fp32(1, 14, 14, 1152):(225792, 16128, 1152, 1):882 KiB
// Elementwise input X_T1279 shape: fp32(1, 14, 14, 1152):(225792, 16128, 1152, 1):882 KiB
// Elementwise input X_I_492 shape: fp32(1152):(1):4.5 KiB
// Elementwise input X_I_491 shape: fp32(1152):(1):4.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1280 = add(X_T1256, X_T1279)
// Elementwise op: [[pid(Sub)]] X_T1282 = sub(X_T1280, X_I_492)
// Elementwise op: [[pid(Mul)]] X_T1283 = mul(X_T1282, X_I_491)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1152):(225792, 16128, 1152, 1):882 KiB
// Computed true ops: 677376
// Computed work groups: 252
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 36, 1
__kernel void kernel_c124_sdk_431(__global float* restrict  X_T1280, __global float* restrict  X_T1283, __global const float* restrict  X_T1256, __global const float* restrict  X_T1279, __global const float* restrict  X_I_492, __global const float* restrict  X_I_491)
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
      int gout_idx = (((16128 * (i2_gid + i2_tid)) + (1152 * i3)) + (i4_gid + i4_tid));
      float LX_T1256 = X_T1256[gout_idx];
      float LX_T1279 = X_T1279[gout_idx];
      float LX_I_492 = X_I_492[(i4_gid + i4_tid)];
      float LX_I_491 = X_I_491[(i4_gid + i4_tid)];
      float LX_T1280 = (LX_T1256 + LX_T1279);
      float LX_T1282 = (LX_T1280 - LX_I_492);
      float LX_T1283 = (LX_T1282 * LX_I_491);
      X_T1280[gout_idx] = LX_T1280;
      X_T1283[gout_idx] = LX_T1283;
    }
  }
}
