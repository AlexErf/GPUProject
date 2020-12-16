#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 352 }
// Out stride: { 275968, 9856, 352, 1 }
// Elementwise input X_T410 shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Elementwise input X_T414 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_159 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T415 = div(X_T410, X_T414)
// Elementwise op: [[pid(Add, Switch)]] X_T416 = add(X_T415, X_I_159)
// Elementwise op: X_T417 = cmp_lt(X_T416, X_T2)
// Elementwise op: [[pid(Relu)]] X_T418 = cond(X_T417, X_T2, X_T416)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Computed true ops: 1103872
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c68_sdk_131(__global float* restrict  X_T418, __global const float* restrict  X_T410, __global const float* restrict  X_T414, __global const float* restrict  X_I_159)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 7; i3_lid += 1)
  {
    int i3 = ((4 * i3_lid) + i3_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((9856 * (i2_gid + i2)) + (352 * i3)) + (i4_gid + i4_tid));
      float LX_T410 = X_T410[gout_idx];
      float LX_T414 = X_T414[(i4_gid + i4_tid)];
      float LX_I_159 = X_I_159[(i4_gid + i4_tid)];
      float LX_T415 = (LX_T410 / LX_T414);
      float LX_T416 = (LX_T415 + LX_I_159);
      int LX_T417 = (LX_T416 < 0.0f);
      float LX_T418 = select((float)LX_T416, (float)0.0f, (int)LX_T417);
      X_T418[gout_idx] = LX_T418;
    }
  }
}
