#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 30 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 960 }
// Out stride: { 188160, 13440, 960, 1 }
// Elementwise input X_T1098 shape: fp32(1, 14, 14, 960):(188160, 13440, 960, 1):735 KiB
// Elementwise input X_T1121 shape: fp32(1, 14, 14, 960):(188160, 13440, 960, 1):735 KiB
// Elementwise input X_I_432 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_431 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1122 = add(X_T1098, X_T1121)
// Elementwise op: [[pid(Sub)]] X_T1124 = sub(X_T1122, X_I_432)
// Elementwise op: [[pid(Mul)]] X_T1125 = mul(X_T1124, X_I_431)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 960):(188160, 13440, 960, 1):735 KiB
// Computed true ops: 564480
// Computed work groups: 210
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 30, 1
__kernel void kernel_c108_sdk_377(__global float* restrict  X_T1122, __global float* restrict  X_T1125, __global const float* restrict  X_T1098, __global const float* restrict  X_T1121, __global const float* restrict  X_I_432, __global const float* restrict  X_I_431)
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
      int gout_idx = (((13440 * (i2_gid + i2_tid)) + (960 * i3)) + (i4_gid + i4_tid));
      float LX_T1098 = X_T1098[gout_idx];
      float LX_T1121 = X_T1121[gout_idx];
      float LX_I_432 = X_I_432[(i4_gid + i4_tid)];
      float LX_I_431 = X_I_431[(i4_gid + i4_tid)];
      float LX_T1122 = (LX_T1098 + LX_T1121);
      float LX_T1124 = (LX_T1122 - LX_I_432);
      float LX_T1125 = (LX_T1124 * LX_I_431);
      X_T1122[gout_idx] = LX_T1122;
      X_T1125[gout_idx] = LX_T1125;
    }
  }
}
