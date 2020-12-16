#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 168 }
// Out stride: { 296352, 7056, 168, 1 }
// Elementwise input X_T463 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T467 shape: fp32(168):(1):672 bytes
// Elementwise input X_I_42 shape: fp32(168):(1):672 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T468 = div(X_T463, X_T467)
// Elementwise op: [[pid(Add, Switch)]] X_T469 = add(X_T468, X_I_42)
// Elementwise op: X_T727 = cmp_lt(X_T469, X_T1)
// Elementwise op: [[pid(Relu)]] X_T728 = cond(X_T727, X_T1, X_T469)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 1185408
// Computed work groups: 1323
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 21, 21
__kernel void kernel_c42_sdk_162(__global float* restrict  X_T469, __global float* restrict  X_T728, __global const float* restrict  X_T463, __global const float* restrict  X_T467, __global const float* restrict  X_I_42)
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
    float LX_T463 = X_T463[gout_idx];
    float LX_T467 = X_T467[(i4_gid + i4_tid)];
    float LX_I_42 = X_I_42[(i4_gid + i4_tid)];
    float LX_T468 = (LX_T463 / LX_T467);
    float LX_T469 = (LX_T468 + LX_I_42);
    int LX_T727 = (LX_T469 < 0.0f);
    float LX_T728 = select((float)LX_T469, (float)0.0f, (int)LX_T727);
    X_T469[gout_idx] = LX_T469;
    X_T728[gout_idx] = LX_T728;
  }
}
