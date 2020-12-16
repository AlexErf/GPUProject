#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 384 }
// Out stride: { 24576, 3072, 384, 1 }
// Elementwise input X_T913 shape: fp32(1, 8, 8, 384):(24576, 3072, 384, 1):96 KiB
// Elementwise input X_T917 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_317 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T918 = div(X_T913, X_T917)
// Elementwise op: [[pid(Add, Switch)]] X_T919 = add(X_T918, X_I_317)
// Elementwise op: X_T920 = cmp_lt(X_T919, X_T2)
// Elementwise op: [[pid(Relu)]] X_T921 = cond(X_T920, X_T2, X_T919)
// Tile size: { 1, 4, 1, 384 }
// Contraction output var shape: fp32(1, 8, 8, 384):(24576, 3072, 384, 1):96 KiB
// Computed true ops: 98304
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 6144
// Computed mem read: 576
// Computed mem write: 6144
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_312(__global float* restrict  X_T921, __global const float* restrict  X_T913, __global const float* restrict  X_T917, __global const float* restrict  X_I_317)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 6; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((3072 * (i2_gid + i2_tid)) + (384 * i3_gid)) + i4);
    float LX_T913 = X_T913[gout_idx];
    float LX_T917 = X_T917[i4];
    float LX_I_317 = X_I_317[i4];
    float LX_T918 = (LX_T913 / LX_T917);
    float LX_T919 = (LX_T918 + LX_I_317);
    int LX_T920 = (LX_T919 < 0.0f);
    float LX_T921 = select((float)LX_T919, (float)0.0f, (int)LX_T920);
    X_T921[gout_idx] = LX_T921;
  }
}
