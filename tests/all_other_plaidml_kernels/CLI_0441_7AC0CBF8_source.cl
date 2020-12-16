#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 35 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1120 }
// Out stride: { 219520, 15680, 1120, 1 }
// Elementwise input X_T1250 shape: fp32(1, 14, 14, 1120):(219520, 15680, 1120, 1):857.5 KiB
// Elementwise input X_T1254 shape: fp32(1120):(1):4.375 KiB
// Elementwise input X_I_480 shape: fp32(1120):(1):4.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1255 = div(X_T1250, X_T1254)
// Elementwise op: [[pid(Add, Switch)]] X_T1256 = add(X_T1255, X_I_480)
// Elementwise op: X_T1257 = cmp_lt(X_T1256, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1258 = cond(X_T1257, X_T2, X_T1256)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1120):(219520, 15680, 1120, 1):857.5 KiB
// Computed true ops: 878080
// Computed work groups: 245
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 35, 1
__kernel void kernel_c108_sdk_425(__global float* restrict  X_T1258, __global const float* restrict  X_T1250, __global const float* restrict  X_T1254, __global const float* restrict  X_I_480)
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
      int gout_idx = (((15680 * (i2_gid + i2_tid)) + (1120 * i3)) + (i4_gid + i4_tid));
      float LX_T1250 = X_T1250[gout_idx];
      float LX_T1254 = X_T1254[(i4_gid + i4_tid)];
      float LX_I_480 = X_I_480[(i4_gid + i4_tid)];
      float LX_T1255 = (LX_T1250 / LX_T1254);
      float LX_T1256 = (LX_T1255 + LX_I_480);
      int LX_T1257 = (LX_T1256 < 0.0f);
      float LX_T1258 = select((float)LX_T1256, (float)0.0f, (int)LX_T1257);
      X_T1258[gout_idx] = LX_T1258;
    }
  }
}
