#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 608 }
// Out stride: { 29792, 4256, 608, 1 }
// Elementwise input X_T1250 shape: fp32(1, 7, 7, 608):(29792, 4256, 608, 1):116.375 KiB
// Elementwise input X_T1254 shape: fp32(608):(1):2.375 KiB
// Elementwise input X_I_481 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1255 = div(X_T1250, X_T1254)
// Elementwise op: [[pid(Add, Switch)]] X_T1256 = add(X_T1255, X_I_481)
// Elementwise op: X_T1257 = cmp_lt(X_T1256, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1258 = cond(X_T1257, X_T2, X_T1256)
// Tile size: { 1, 1, 1, 608 }
// Contraction output var shape: fp32(1, 7, 7, 608):(29792, 4256, 608, 1):116.375 KiB
// Computed true ops: 119168
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 228
// Computed mem write: 2432
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_431(__global float* restrict  X_T1258, __global const float* restrict  X_T1250, __global const float* restrict  X_T1254, __global const float* restrict  X_I_481)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((4256 * i2_gid) + (608 * i3_gid)) + i4);
      float LX_T1250 = X_T1250[gout_idx];
      float LX_T1254 = X_T1254[i4];
      float LX_I_481 = X_I_481[i4];
      float LX_T1255 = (LX_T1250 / LX_T1254);
      float LX_T1256 = (LX_T1255 + LX_I_481);
      int LX_T1257 = (LX_T1256 < 0.0f);
      float LX_T1258 = select((float)LX_T1256, (float)0.0f, (int)LX_T1257);
      X_T1258[gout_idx] = LX_T1258;
    }
  }
}
