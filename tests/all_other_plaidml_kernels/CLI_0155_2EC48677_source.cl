#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 168 }
// Out stride: { 296352, 7056, 168, 1 }
// Elementwise input X_T527 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T531 shape: fp32(168):(1):672 bytes
// Elementwise input X_I_214 shape: fp32(168):(1):672 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T532 = div(X_T527, X_T531)
// Elementwise op: [[pid(Add, Switch)]] X_T533 = add(X_T532, X_I_214)
// Elementwise op: X_T534 = cmp_lt(X_T533, X_T1)
// Elementwise op: [[pid(Relu)]] X_T535 = cond(X_T534, X_T1, X_T533)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 1185408
// Computed work groups: 1323
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 21
__kernel void kernel_c42_sdk_190(__global float* restrict  X_T535, __global const float* restrict  X_T527, __global const float* restrict  X_T531, __global const float* restrict  X_I_214)
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
    float LX_T527 = X_T527[gout_idx];
    float LX_T531 = X_T531[(i4_gid + i4_tid)];
    float LX_I_214 = X_I_214[(i4_gid + i4_tid)];
    float LX_T532 = (LX_T527 / LX_T531);
    float LX_T533 = (LX_T532 + LX_I_214);
    int LX_T534 = (LX_T533 < 0.0f);
    float LX_T535 = select((float)LX_T533, (float)0.0f, (int)LX_T534);
    X_T535[gout_idx] = LX_T535;
  }
}
