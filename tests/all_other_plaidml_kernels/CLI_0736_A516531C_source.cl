#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1792 }
// Out stride: { 87808, 12544, 1792, 1 }
// Elementwise input X_T2476 shape: fp32(1, 7, 7, 1792):(87808, 12544, 1792, 1):343 KiB
// Elementwise input X_T2499 shape: fp32(1, 7, 7, 1792):(87808, 12544, 1792, 1):343 KiB
// Elementwise input X_I_973 shape: fp32(1792):(1):7 KiB
// Elementwise input X_I_972 shape: fp32(1792):(1):7 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2500 = add(X_T2476, X_T2499)
// Elementwise op: [[pid(Sub)]] X_T2502 = sub(X_T2500, X_I_973)
// Elementwise op: [[pid(Mul)]] X_T2503 = mul(X_T2502, X_I_972)
// Tile size: { 1, 1, 1, 1792 }
// Contraction output var shape: fp32(1, 7, 7, 1792):(87808, 12544, 1792, 1):343 KiB
// Computed true ops: 263424
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_869(__global float* restrict  X_T2500, __global float* restrict  X_T2503, __global const float* restrict  X_T2476, __global const float* restrict  X_T2499, __global const float* restrict  X_I_973, __global const float* restrict  X_I_972)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((256 * i4_lid) + i4_tid);
    int gout_idx = (((12544 * i2_gid) + (1792 * i3_gid)) + i4);
    float LX_T2476 = X_T2476[gout_idx];
    float LX_T2499 = X_T2499[gout_idx];
    float LX_I_973 = X_I_973[i4];
    float LX_I_972 = X_I_972[i4];
    float LX_T2500 = (LX_T2476 + LX_T2499);
    float LX_T2502 = (LX_T2500 - LX_I_973);
    float LX_T2503 = (LX_T2502 * LX_I_972);
    X_T2500[gout_idx] = LX_T2500;
    X_T2503[gout_idx] = LX_T2503;
  }
}
