#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 44 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1408 }
// Out stride: { 275968, 19712, 1408, 1 }
// Elementwise input X_T1456 shape: fp32(1, 14, 14, 1408):(275968, 19712, 1408, 1):1078 KiB
// Elementwise input X_T1479 shape: fp32(1, 14, 14, 1408):(275968, 19712, 1408, 1):1078 KiB
// Elementwise input X_I_572 shape: fp32(1408):(1):5.5 KiB
// Elementwise input X_I_571 shape: fp32(1408):(1):5.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1480 = add(X_T1456, X_T1479)
// Elementwise op: [[pid(Sub)]] X_T1482 = sub(X_T1480, X_I_572)
// Elementwise op: [[pid(Mul)]] X_T1483 = mul(X_T1482, X_I_571)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1408):(275968, 19712, 1408, 1):1078 KiB
// Computed true ops: 827904
// Computed work groups: 308
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 44, 1
__kernel void kernel_c124_sdk_503(__global float* restrict  X_T1480, __global float* restrict  X_T1483, __global const float* restrict  X_T1456, __global const float* restrict  X_T1479, __global const float* restrict  X_I_572, __global const float* restrict  X_I_571)
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
      int gout_idx = (((19712 * (i2_gid + i2_tid)) + (1408 * i3)) + (i4_gid + i4_tid));
      float LX_T1456 = X_T1456[gout_idx];
      float LX_T1479 = X_T1479[gout_idx];
      float LX_I_572 = X_I_572[(i4_gid + i4_tid)];
      float LX_I_571 = X_I_571[(i4_gid + i4_tid)];
      float LX_T1480 = (LX_T1456 + LX_T1479);
      float LX_T1482 = (LX_T1480 - LX_I_572);
      float LX_T1483 = (LX_T1482 * LX_I_571);
      X_T1480[gout_idx] = LX_T1480;
      X_T1483[gout_idx] = LX_T1483;
    }
  }
}
