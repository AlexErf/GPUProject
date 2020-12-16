#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 19 19
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 19, 19, 728 }
// Out stride: { 262808, 13832, 728, 1 }
// Elementwise input X_T238 shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Elementwise input X_T242 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_I_199 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_T231 shape: fp32(1, 19, 19, 728):(262808, 13832, 728, 1):1026.59 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T243 = div(X_T238, X_T242)
// Elementwise op: [[pid(Add, Switch)]] X_T244 = add(X_T243, X_I_199)
// Elementwise op: [[pid(Add)]] X_T245 = add(X_T231, X_T244)
// Elementwise op: X_T246 = cmp_lt(X_T245, X_T2)
// Elementwise op: [[pid(Relu)]] X_T247 = cond(X_T246, X_T2, X_T245)
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
__kernel void kernel_c28_sdk_75(__global float* restrict  X_T245, __global float* restrict  X_T247, __global const float* restrict  X_T238, __global const float* restrict  X_T242, __global const float* restrict  X_I_199, __global const float* restrict  X_T231)
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
    float LX_T238 = X_T238[gout_idx];
    float LX_T242 = X_T242[(i4_gid + i4_tid)];
    float LX_I_199 = X_I_199[(i4_gid + i4_tid)];
    float LX_T231 = X_T231[gout_idx];
    float LX_T243 = (LX_T238 / LX_T242);
    float LX_T244 = (LX_T243 + LX_I_199);
    float LX_T245 = (LX_T231 + LX_T244);
    int LX_T246 = (LX_T245 < 0.0f);
    float LX_T247 = select((float)LX_T245, (float)0.0f, (int)LX_T246);
    X_T245[gout_idx] = LX_T245;
    X_T247[gout_idx] = LX_T247;
  }
}
