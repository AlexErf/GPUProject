#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 896 }
// Out stride: { 175616, 12544, 896, 1 }
// Elementwise input X_T1048 shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Elementwise input X_T1071 shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Elementwise input X_I_412 shape: fp32(896):(1):3.5 KiB
// Elementwise input X_I_411 shape: fp32(896):(1):3.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1072 = add(X_T1048, X_T1071)
// Elementwise op: [[pid(Sub)]] X_T1074 = sub(X_T1072, X_I_412)
// Elementwise op: [[pid(Mul)]] X_T1075 = mul(X_T1074, X_I_411)
// Tile size: { 1, 2, 2, 896 }
// Contraction output var shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Computed true ops: 526848
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_359(__global float* restrict  X_T1072, __global float* restrict  X_T1075, __global const float* restrict  X_T1048, __global const float* restrict  X_T1071, __global const float* restrict  X_I_412, __global const float* restrict  X_I_411)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 14; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((12544 * (i2_gid + i2_tid)) + (896 * (i3_gid + i3_tid))) + i4);
    float LX_T1048 = X_T1048[gout_idx];
    float LX_T1071 = X_T1071[gout_idx];
    float LX_I_412 = X_I_412[i4];
    float LX_I_411 = X_I_411[i4];
    float LX_T1072 = (LX_T1048 + LX_T1071);
    float LX_T1074 = (LX_T1072 - LX_I_412);
    float LX_T1075 = (LX_T1074 * LX_I_411);
    X_T1072[gout_idx] = LX_T1072;
    X_T1075[gout_idx] = LX_T1075;
  }
}
