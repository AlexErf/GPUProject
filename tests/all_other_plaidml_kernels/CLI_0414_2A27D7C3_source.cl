#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 31 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 992 }
// Out stride: { 194432, 13888, 992, 1 }
// Elementwise input X_T1123 shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Elementwise input X_T1146 shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Elementwise input X_I_442 shape: fp32(992):(1):3.875 KiB
// Elementwise input X_I_441 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1147 = add(X_T1123, X_T1146)
// Elementwise op: [[pid(Sub)]] X_T1149 = sub(X_T1147, X_I_442)
// Elementwise op: [[pid(Mul)]] X_T1150 = mul(X_T1149, X_I_441)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Computed true ops: 583296
// Computed work groups: 217
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 31, 1
__kernel void kernel_c108_sdk_386(__global float* restrict  X_T1147, __global float* restrict  X_T1150, __global const float* restrict  X_T1123, __global const float* restrict  X_T1146, __global const float* restrict  X_I_442, __global const float* restrict  X_I_441)
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
      int gout_idx = (((13888 * (i2_gid + i2_tid)) + (992 * i3)) + (i4_gid + i4_tid));
      float LX_T1123 = X_T1123[gout_idx];
      float LX_T1146 = X_T1146[gout_idx];
      float LX_I_442 = X_I_442[(i4_gid + i4_tid)];
      float LX_I_441 = X_I_441[(i4_gid + i4_tid)];
      float LX_T1147 = (LX_T1123 + LX_T1146);
      float LX_T1149 = (LX_T1147 - LX_I_442);
      float LX_T1150 = (LX_T1149 * LX_I_441);
      X_T1147[gout_idx] = LX_T1147;
      X_T1150[gout_idx] = LX_T1150;
    }
  }
}
