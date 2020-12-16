#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 576 }
// Out stride: { 28224, 4032, 576, 1 }
// Elementwise input X_T1225 shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Elementwise input X_T1229 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_471 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1230 = div(X_T1225, X_T1229)
// Elementwise op: [[pid(Add, Switch)]] X_T1231 = add(X_T1230, X_I_471)
// Elementwise op: X_T1232 = cmp_lt(X_T1231, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1233 = cond(X_T1232, X_T2, X_T1231)
// Tile size: { 1, 1, 1, 576 }
// Contraction output var shape: fp32(1, 7, 7, 576):(28224, 4032, 576, 1):110.25 KiB
// Computed true ops: 112896
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
__kernel void kernel_c68_sdk_422(__global float* restrict  X_T1233, __global const float* restrict  X_T1225, __global const float* restrict  X_T1229, __global const float* restrict  X_I_471)
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
      float LX_T1225 = X_T1225[gout_idx];
      float LX_T1229 = X_T1229[i4];
      float LX_I_471 = X_I_471[i4];
      float LX_T1230 = (LX_T1225 / LX_T1229);
      float LX_T1231 = (LX_T1230 + LX_I_471);
      int LX_T1232 = (LX_T1231 < 0.0f);
      float LX_T1233 = select((float)LX_T1231, (float)0.0f, (int)LX_T1232);
      X_T1233[gout_idx] = LX_T1233;
    }
  }
}
