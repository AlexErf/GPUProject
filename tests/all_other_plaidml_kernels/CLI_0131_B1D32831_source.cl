#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 192 }
// Out stride: { 602112, 10752, 192, 1 }
// Elementwise input X_T137 shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Elementwise input X_T160 shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Elementwise input X_I_70 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_69 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Concatenate)]] X_T161 = add(X_T137, X_T160)
// Elementwise op: [[pid(Sub)]] X_T163 = sub(X_T161, X_I_70)
// Elementwise op: [[pid(Mul)]] X_T164 = mul(X_T163, X_I_69)
// Tile size: { 1, 56, 2, 32 }
// Contraction output var shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Computed true ops: 1806336
// Computed work groups: 168
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 28, 1
__kernel void kernel_c68_sdk_41(__global float* restrict  X_T161, __global float* restrict  X_T164, __global const float* restrict  X_T137, __global const float* restrict  X_T160, __global const float* restrict  X_I_70, __global const float* restrict  X_I_69)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
  {
    int i2 = ((4 * i2_lid) + i2_tid);
    int gout_idx = (((10752 * i2) + (192 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T137 = X_T137[gout_idx];
    float LX_T160 = X_T160[gout_idx];
    float LX_I_70 = X_I_70[(i4_gid + i4_tid)];
    float LX_I_69 = X_I_69[(i4_gid + i4_tid)];
    float LX_T161 = (LX_T137 + LX_T160);
    float LX_T163 = (LX_T161 - LX_I_70);
    float LX_T164 = (LX_T163 * LX_I_69);
    X_T161[gout_idx] = LX_T161;
    X_T164[gout_idx] = LX_T164;
  }
}
