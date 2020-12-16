#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1568 }
// Out stride: { 76832, 10976, 1568, 1 }
// Elementwise input X_T2120 shape: fp32(1, 7, 7, 1568):(76832, 10976, 1568, 1):300.125 KiB
// Elementwise input X_T2124 shape: fp32(1568):(1):6.125 KiB
// Elementwise input X_I_821 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2125 = div(X_T2120, X_T2124)
// Elementwise op: [[pid(Add, Switch)]] X_T2126 = add(X_T2125, X_I_821)
// Elementwise op: X_T2127 = cmp_lt(X_T2126, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2128 = cond(X_T2127, X_T2, X_T2126)
// Tile size: { 1, 1, 1, 1568 }
// Contraction output var shape: fp32(1, 7, 7, 1568):(76832, 10976, 1568, 1):300.125 KiB
// Computed true ops: 307328
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 588
// Computed mem write: 6272
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_737(__global float* restrict  X_T2128, __global const float* restrict  X_T2120, __global const float* restrict  X_T2124, __global const float* restrict  X_I_821)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((10976 * i2_gid) + (1568 * i3_gid)) + i4);
      float LX_T2120 = X_T2120[gout_idx];
      float LX_T2124 = X_T2124[i4];
      float LX_I_821 = X_I_821[i4];
      float LX_T2125 = (LX_T2120 / LX_T2124);
      float LX_T2126 = (LX_T2125 + LX_I_821);
      int LX_T2127 = (LX_T2126 < 0.0f);
      float LX_T2128 = select((float)LX_T2126, (float)0.0f, (int)LX_T2127);
      X_T2128[gout_idx] = LX_T2128;
    }
  }
}
