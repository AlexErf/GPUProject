#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1632 }
// Out stride: { 79968, 11424, 1632, 1 }
// Elementwise input X_T2378 shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Elementwise input X_T2382 shape: fp32(1632):(1):6.375 KiB
// Elementwise input X_I_921 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2383 = div(X_T2378, X_T2382)
// Elementwise op: [[pid(Add, Switch)]] X_T2384 = add(X_T2383, X_I_921)
// Elementwise op: X_T2385 = cmp_lt(X_T2384, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2386 = cond(X_T2385, X_T2, X_T2384)
// Tile size: { 1, 1, 1, 1632 }
// Contraction output var shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Computed true ops: 319872
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 612
// Computed mem write: 6528
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_827(__global float* restrict  X_T2386, __global const float* restrict  X_T2378, __global const float* restrict  X_T2382, __global const float* restrict  X_I_921)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11424 * i2_gid) + (1632 * i3_gid)) + i4);
      float LX_T2378 = X_T2378[gout_idx];
      float LX_T2382 = X_T2382[i4];
      float LX_I_921 = X_I_921[i4];
      float LX_T2383 = (LX_T2378 / LX_T2382);
      float LX_T2384 = (LX_T2383 + LX_I_921);
      int LX_T2385 = (LX_T2384 < 0.0f);
      float LX_T2386 = select((float)LX_T2384, (float)0.0f, (int)LX_T2385);
      X_T2386[gout_idx] = LX_T2386;
    }
  }
}
