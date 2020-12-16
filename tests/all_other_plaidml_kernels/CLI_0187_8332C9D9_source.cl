#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 352 }
// Out stride: { 275968, 9856, 352, 1 }
// Elementwise input X_T383 shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Elementwise input X_T406 shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Elementwise input X_I_161 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_160 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T407 = add(X_T383, X_T406)
// Elementwise op: [[pid(Sub)]] X_T409 = sub(X_T407, X_I_161)
// Elementwise op: [[pid(Mul)]] X_T410 = mul(X_T409, X_I_160)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Computed true ops: 827904
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c68_sdk_128(__global float* restrict  X_T407, __global float* restrict  X_T410, __global const float* restrict  X_T383, __global const float* restrict  X_T406, __global const float* restrict  X_I_161, __global const float* restrict  X_I_160)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 7; i3_lid += 1)
  {
    int i3 = ((4 * i3_lid) + i3_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((9856 * (i2_gid + i2)) + (352 * i3)) + (i4_gid + i4_tid));
      float LX_T383 = X_T383[gout_idx];
      float LX_T406 = X_T406[gout_idx];
      float LX_I_161 = X_I_161[(i4_gid + i4_tid)];
      float LX_I_160 = X_I_160[(i4_gid + i4_tid)];
      float LX_T407 = (LX_T383 + LX_T406);
      float LX_T409 = (LX_T407 - LX_I_161);
      float LX_T410 = (LX_T409 * LX_I_160);
      X_T407[gout_idx] = LX_T407;
      X_T410[gout_idx] = LX_T410;
    }
  }
}
