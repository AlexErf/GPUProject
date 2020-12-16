#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 576 }
// Out stride: { 28224, 4032, 576, 1 }
// Elementwise input X_T573 shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Elementwise input X_T577 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_22 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T578 = div(X_T573, X_T577)
// Elementwise op: [[pid(Add, Switch)]] X_T579 = add(X_T578, X_I_22)
// Elementwise op: X_T580 = cmp_lt(X_T579, X_T3)
// Elementwise op: [[pid(Relu)]] X_T581 = cond(X_T580, X_T3, X_T579)
// Elementwise op: X_T582 = cmp_lt(X_T581, X_T2)
// Elementwise op: [[pid(Relu)]] X_T583 = cond(X_T582, X_T581, X_T2)
// Tile size: { 1, 1, 1, 576 }
// Contraction output var shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Computed true ops: 169344
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 216
// Computed mem write: 2304
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_156(__global float* restrict  X_T583, __global const float* restrict  X_T573, __global const float* restrict  X_T577, __global const float* restrict  X_I_22)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4032 * i2_gid) + (576 * i3_gid)) + i4);
      float LX_T573 = X_T573[gout_idx];
      float LX_T577 = X_T577[i4];
      float LX_I_22 = X_I_22[i4];
      float LX_T578 = (LX_T573 / LX_T577);
      float LX_T579 = (LX_T578 + LX_I_22);
      int LX_T580 = (LX_T579 < 0.0f);
      float LX_T581 = select((float)LX_T579, (float)0.0f, (int)LX_T580);
      int LX_T582 = (LX_T581 < 6.0f);
      float LX_T583 = select((float)6.0f, (float)LX_T581, (int)LX_T582);
      X_T583[gout_idx] = LX_T583;
    }
  }
}
