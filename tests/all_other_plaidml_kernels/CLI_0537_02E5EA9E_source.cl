#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 47 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1504 }
// Out stride: { 294784, 21056, 1504, 1 }
// Elementwise input X_T1558 shape: fp32(1, 14, 14, 1504):(294784, 21056, 1504, 1):1151.5 KiB
// Elementwise input X_T1562 shape: fp32(1504):(1):5.875 KiB
// Elementwise input X_I_600 shape: fp32(1504):(1):5.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1563 = div(X_T1558, X_T1562)
// Elementwise op: [[pid(Add, Switch)]] X_T1564 = add(X_T1563, X_I_600)
// Elementwise op: X_T1565 = cmp_lt(X_T1564, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1566 = cond(X_T1565, X_T2, X_T1564)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1504):(294784, 21056, 1504, 1):1151.5 KiB
// Computed true ops: 1179136
// Computed work groups: 329
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 47, 1
__kernel void kernel_c124_sdk_533(__global float* restrict  X_T1566, __global const float* restrict  X_T1558, __global const float* restrict  X_T1562, __global const float* restrict  X_I_600)
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
      int gout_idx = (((21056 * (i2_gid + i2_tid)) + (1504 * i3)) + (i4_gid + i4_tid));
      float LX_T1558 = X_T1558[gout_idx];
      float LX_T1562 = X_T1562[(i4_gid + i4_tid)];
      float LX_I_600 = X_I_600[(i4_gid + i4_tid)];
      float LX_T1563 = (LX_T1558 / LX_T1562);
      float LX_T1564 = (LX_T1563 + LX_I_600);
      int LX_T1565 = (LX_T1564 < 0.0f);
      float LX_T1566 = select((float)LX_T1564, (float)0.0f, (int)LX_T1565);
      X_T1566[gout_idx] = LX_T1566;
    }
  }
}
