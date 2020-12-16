#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1600 }
// Out stride: { 78400, 11200, 1600, 1 }
// Elementwise input X_T2353 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_T2357 shape: fp32(1600):(1):6.25 KiB
// Elementwise input X_I_911 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2358 = div(X_T2353, X_T2357)
// Elementwise op: [[pid(Add, Switch)]] X_T2359 = add(X_T2358, X_I_911)
// Elementwise op: X_T2360 = cmp_lt(X_T2359, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2361 = cond(X_T2360, X_T2, X_T2359)
// Tile size: { 1, 1, 1, 1600 }
// Contraction output var shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Computed true ops: 313600
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 600
// Computed mem write: 6400
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_818(__global float* restrict  X_T2361, __global const float* restrict  X_T2353, __global const float* restrict  X_T2357, __global const float* restrict  X_I_911)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11200 * i2_gid) + (1600 * i3_gid)) + i4);
      float LX_T2353 = X_T2353[gout_idx];
      float LX_T2357 = X_T2357[i4];
      float LX_I_911 = X_I_911[i4];
      float LX_T2358 = (LX_T2353 / LX_T2357);
      float LX_T2359 = (LX_T2358 + LX_I_911);
      int LX_T2360 = (LX_T2359 < 0.0f);
      float LX_T2361 = select((float)LX_T2359, (float)0.0f, (int)LX_T2360);
      X_T2361[gout_idx] = LX_T2361;
    }
  }
}
