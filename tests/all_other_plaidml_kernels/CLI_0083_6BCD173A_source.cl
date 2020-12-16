#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 19 19
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 19, 19, 728 }
// Out stride: { 262808, 13832, 728, 1 }
// Elementwise input X_T252 shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Elementwise input X_T256 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_I_137 shape: fp32(728):(1):2.84375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T257 = div(X_T252, X_T256)
// Elementwise op: [[pid(Add, Switch)]] X_T258 = add(X_T257, X_I_137)
// Elementwise op: X_T259 = cmp_lt(X_T258, X_T2)
// Elementwise op: [[pid(Relu)]] X_T260 = cond(X_T259, X_T2, X_T258)
// Tile size: { 1, 1, 1, 256 }
// Contraction output var shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Computed true ops: 1051232
// Computed work groups: 1083
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 19, 19
__kernel void kernel_c28_sdk_79(__global float* restrict  X_T260, __global const float* restrict  X_T252, __global const float* restrict  X_T256, __global const float* restrict  X_I_137)
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
    float LX_T252 = X_T252[gout_idx];
    float LX_T256 = X_T256[(i4_gid + i4_tid)];
    float LX_I_137 = X_I_137[(i4_gid + i4_tid)];
    float LX_T257 = (LX_T252 / LX_T256);
    float LX_T258 = (LX_T257 + LX_I_137);
    int LX_T259 = (LX_T258 < 0.0f);
    float LX_T260 = select((float)LX_T258, (float)0.0f, (int)LX_T259);
    X_T260[gout_idx] = LX_T260;
  }
}
