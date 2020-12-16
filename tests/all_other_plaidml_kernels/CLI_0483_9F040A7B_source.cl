#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 672 }
// Out stride: { 32928, 4704, 672, 1 }
// Elementwise input X_T1420 shape: fp32(1, 7, 7, 672):(32928, 4704, 672, 1):128.625 KiB
// Elementwise input X_T1424 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_541 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1425 = div(X_T1420, X_T1424)
// Elementwise op: [[pid(Add, Switch)]] X_T1426 = add(X_T1425, X_I_541)
// Elementwise op: X_T1427 = cmp_lt(X_T1426, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1428 = cond(X_T1427, X_T2, X_T1426)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 672):(32928, 4704, 672, 1):128.625 KiB
// Computed true ops: 131712
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c108_sdk_485(__global float* restrict  X_T1428, __global const float* restrict  X_T1420, __global const float* restrict  X_T1424, __global const float* restrict  X_I_541)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 640));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((4704 * i2_gid) + (672 * i3_tid)) + (i4_gid + i4));
        float LX_T1420 = X_T1420[gout_idx];
        float LX_T1424 = X_T1424[(i4_gid + i4)];
        float LX_I_541 = X_I_541[(i4_gid + i4)];
        float LX_T1425 = (LX_T1420 / LX_T1424);
        float LX_T1426 = (LX_T1425 + LX_I_541);
        int LX_T1427 = (LX_T1426 < 0.0f);
        float LX_T1428 = select((float)LX_T1426, (float)0.0f, (int)LX_T1427);
        X_T1428[gout_idx] = LX_T1428;
      }
    }
  }
}
