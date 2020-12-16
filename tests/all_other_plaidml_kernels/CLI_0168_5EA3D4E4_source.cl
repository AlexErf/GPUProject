#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 168 }
// Out stride: { 296352, 7056, 168, 1 }
// Elementwise input X_T678 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T682 shape: fp32(168):(1):672 bytes
// Elementwise input X_I_262 shape: fp32(168):(1):672 bytes
// Elementwise input X_T520 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T683 = div(X_T678, X_T682)
// Elementwise op: [[pid(Add, Switch)]] X_T684 = add(X_T683, X_I_262)
// Elementwise op: [[pid(Add)]] X_T685 = add(X_T684, X_T520)
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
__kernel void kernel_c42_sdk_248(__global float* restrict  X_T685, __global const float* restrict  X_T678, __global const float* restrict  X_T682, __global const float* restrict  X_I_262, __global const float* restrict  X_T520)
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
    float LX_T678 = X_T678[gout_idx];
    float LX_T682 = X_T682[(i4_gid + i4_tid)];
    float LX_I_262 = X_I_262[(i4_gid + i4_tid)];
    float LX_T520 = X_T520[gout_idx];
    float LX_T683 = (LX_T678 / LX_T682);
    float LX_T684 = (LX_T683 + LX_I_262);
    float LX_T685 = (LX_T684 + LX_T520);
    X_T685[gout_idx] = LX_T685;
  }
}
