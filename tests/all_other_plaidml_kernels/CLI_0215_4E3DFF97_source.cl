#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 192 }
// Out stride: { 602112, 10752, 192, 1 }
// Elementwise input X_T165 shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Elementwise input X_T188 shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Elementwise input X_I_70 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_69 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Concatenate)]] X_T189 = add(X_T165, X_T188)
// Elementwise op: [[pid(Sub)]] X_T191 = sub(X_T189, X_I_70)
// Elementwise op: [[pid(Mul)]] X_T192 = mul(X_T191, X_I_69)
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
__kernel void kernel_c124_sdk_41(__global float* restrict  X_T189, __global float* restrict  X_T192, __global const float* restrict  X_T165, __global const float* restrict  X_T188, __global const float* restrict  X_I_70, __global const float* restrict  X_I_69)
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
    float LX_T165 = X_T165[gout_idx];
    float LX_T188 = X_T188[gout_idx];
    float LX_I_70 = X_I_70[(i4_gid + i4_tid)];
    float LX_I_69 = X_I_69[(i4_gid + i4_tid)];
    float LX_T189 = (LX_T165 + LX_T188);
    float LX_T191 = (LX_T189 - LX_I_70);
    float LX_T192 = (LX_T191 * LX_I_69);
    X_T189[gout_idx] = LX_T189;
    X_T192[gout_idx] = LX_T192;
  }
}
