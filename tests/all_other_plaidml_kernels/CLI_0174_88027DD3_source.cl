#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 21 21
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 168 }
// Out stride: { 296352, 7056, 168, 1 }
// Elementwise input X_T854 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise input X_T858 shape: fp32(168):(1):672 bytes
// Elementwise input X_I_38 shape: fp32(168):(1):672 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T859 = div(X_T854, X_T858)
// Elementwise op: [[pid(Add, Switch)]] X_T860 = add(X_T859, X_I_38)
// Elementwise op: X_T1075 = cmp_lt(X_T860, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1076 = cond(X_T1075, X_T1, X_T860)
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
__kernel void kernel_c42_sdk_316(__global float* restrict  X_T1076, __global float* restrict  X_T860, __global const float* restrict  X_T854, __global const float* restrict  X_T858, __global const float* restrict  X_I_38)
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
    float LX_T854 = X_T854[gout_idx];
    float LX_T858 = X_T858[(i4_gid + i4_tid)];
    float LX_I_38 = X_I_38[(i4_gid + i4_tid)];
    float LX_T859 = (LX_T854 / LX_T858);
    float LX_T860 = (LX_T859 + LX_I_38);
    int LX_T1075 = (LX_T860 < 0.0f);
    float LX_T1076 = select((float)LX_T860, (float)0.0f, (int)LX_T1075);
    X_T1076[gout_idx] = LX_T1076;
    X_T860[gout_idx] = LX_T860;
  }
}
