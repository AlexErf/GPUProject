#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 19 19
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 19, 19, 728 }
// Out stride: { 262808, 13832, 728, 1 }
// Elementwise input X_T278 shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Elementwise input X_T282 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_I_127 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_T245 shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T283 = div(X_T278, X_T282)
// Elementwise op: [[pid(Add, Switch)]] X_T284 = add(X_T283, X_I_127)
// Elementwise op: [[pid(Add)]] X_T285 = add(X_T284, X_T245)
// Elementwise op: X_T286 = cmp_lt(X_T285, X_T2)
// Elementwise op: [[pid(Relu)]] X_T287 = cond(X_T286, X_T2, X_T285)
// Tile size: { 1, 1, 1, 256 }
// Contraction output var shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Computed true ops: 1314040
// Computed work groups: 1083
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 19, 19
__kernel void kernel_c28_sdk_87(__global float* restrict  X_T285, __global float* restrict  X_T287, __global const float* restrict  X_T278, __global const float* restrict  X_T282, __global const float* restrict  X_I_127, __global const float* restrict  X_T245)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 256);
  int i3_gid = get_group_id(1);
  int i2_gid = get_group_id(2);
  int i4_tid = (tid % 256);
  int i4_cond = ((i4_gid != 512) || (i4_tid < 216));
  if (i4_cond)
  {
    int gout_idx = (((13832 * i2_gid) + (728 * i3_gid)) + (i4_gid + i4_tid));
    float LX_T278 = X_T278[gout_idx];
    float LX_T282 = X_T282[(i4_gid + i4_tid)];
    float LX_I_127 = X_I_127[(i4_gid + i4_tid)];
    float LX_T245 = X_T245[gout_idx];
    float LX_T283 = (LX_T278 / LX_T282);
    float LX_T284 = (LX_T283 + LX_I_127);
    float LX_T285 = (LX_T284 + LX_T245);
    int LX_T286 = (LX_T285 < 0.0f);
    float LX_T287 = select((float)LX_T285, (float)0.0f, (int)LX_T286);
    X_T285[gout_idx] = LX_T285;
    X_T287[gout_idx] = LX_T287;
  }
}
