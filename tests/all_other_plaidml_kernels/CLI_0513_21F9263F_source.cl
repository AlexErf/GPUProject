#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 43 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1376 }
// Out stride: { 269696, 19264, 1376, 1 }
// Elementwise input X_T1458 shape: fp32(1, 14, 14, 1376):(269696, 19264, 1376, 1):1053.5 KiB
// Elementwise input X_T1462 shape: fp32(1376):(1):5.375 KiB
// Elementwise input X_I_560 shape: fp32(1376):(1):5.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1463 = div(X_T1458, X_T1462)
// Elementwise op: [[pid(Add, Switch)]] X_T1464 = add(X_T1463, X_I_560)
// Elementwise op: X_T1465 = cmp_lt(X_T1464, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1466 = cond(X_T1465, X_T2, X_T1464)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1376):(269696, 19264, 1376, 1):1053.5 KiB
// Computed true ops: 1078784
// Computed work groups: 301
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 43, 1
__kernel void kernel_c124_sdk_497(__global float* restrict  X_T1466, __global const float* restrict  X_T1458, __global const float* restrict  X_T1462, __global const float* restrict  X_I_560)
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
      int gout_idx = (((19264 * (i2_gid + i2_tid)) + (1376 * i3)) + (i4_gid + i4_tid));
      float LX_T1458 = X_T1458[gout_idx];
      float LX_T1462 = X_T1462[(i4_gid + i4_tid)];
      float LX_I_560 = X_I_560[(i4_gid + i4_tid)];
      float LX_T1463 = (LX_T1458 / LX_T1462);
      float LX_T1464 = (LX_T1463 + LX_I_560);
      int LX_T1465 = (LX_T1464 < 0.0f);
      float LX_T1466 = select((float)LX_T1464, (float)0.0f, (int)LX_T1465);
      X_T1466[gout_idx] = LX_T1466;
    }
  }
}
