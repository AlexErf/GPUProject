#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 23 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 736 }
// Out stride: { 144256, 10304, 736, 1 }
// Elementwise input X_T930 shape: fp32(1, 14, 14, 736):(144256, 10304, 736, 1):563.5 KiB
// Elementwise input X_T934 shape: fp32(736):(1):2.875 KiB
// Elementwise input X_I_360 shape: fp32(736):(1):2.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T935 = div(X_T930, X_T934)
// Elementwise op: [[pid(Add, Switch)]] X_T936 = add(X_T935, X_I_360)
// Elementwise op: X_T937 = cmp_lt(X_T936, X_T2)
// Elementwise op: [[pid(Relu)]] X_T938 = cond(X_T937, X_T2, X_T936)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 736):(144256, 10304, 736, 1):563.5 KiB
// Computed true ops: 577024
// Computed work groups: 161
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 23, 1
__kernel void kernel_c68_sdk_317(__global float* restrict  X_T938, __global const float* restrict  X_T930, __global const float* restrict  X_T934, __global const float* restrict  X_I_360)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((10304 * (i2_gid + i2_tid)) + (736 * i3)) + (i4_gid + i4_tid));
      float LX_T930 = X_T930[gout_idx];
      float LX_T934 = X_T934[(i4_gid + i4_tid)];
      float LX_I_360 = X_I_360[(i4_gid + i4_tid)];
      float LX_T935 = (LX_T930 / LX_T934);
      float LX_T936 = (LX_T935 + LX_I_360);
      int LX_T937 = (LX_T936 < 0.0f);
      float LX_T938 = select((float)LX_T936, (float)0.0f, (int)LX_T937);
      X_T938[gout_idx] = LX_T938;
    }
  }
}
