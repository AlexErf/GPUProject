#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 168 }
// Out stride: { 296352, 7056, 168, 1 }
// Elementwise input X_T487 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T499 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_I_205 shape: fp32(168):(1):672 bytes
// Elementwise input X_I_204 shape: fp32(168):(1):672 bytes
// Elementwise op: [[pid(Concatenate)]] X_T500 = add(X_T487, X_T499)
// Elementwise op: [[pid(Sub)]] X_T501 = sub(X_T500, X_I_205)
// Elementwise op: [[pid(Mul)]] X_T502 = mul(X_T501, X_I_204)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 889056
// Computed work groups: 1323
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 21
__kernel void kernel_c42_sdk_178(__global float* restrict  X_T502, __global const float* restrict  X_T487, __global const float* restrict  X_T499, __global const float* restrict  X_I_205, __global const float* restrict  X_I_204)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i4_cond = ((i4_gid != 128) || (i4_tid < 40));
  if (i4_cond)
  {
    int gout_idx = (((7056 * (i2_gid + i2_tid)) + (168 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T487 = X_T487[gout_idx];
    float LX_T499 = X_T499[gout_idx];
    float LX_I_205 = X_I_205[(i4_gid + i4_tid)];
    float LX_I_204 = X_I_204[(i4_gid + i4_tid)];
    float LX_T500 = (LX_T487 + LX_T499);
    float LX_T501 = (LX_T500 - LX_I_205);
    float LX_T502 = (LX_T501 * LX_I_204);
    X_T502[gout_idx] = LX_T502;
  }
}
