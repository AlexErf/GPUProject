#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 352 }
// Out stride: { 275968, 9856, 352, 1 }
// Elementwise input X_T430 shape: fp32(1, 28, 28, 352):(275968, 9856, 352, 1):1078 KiB
// Elementwise input X_T434 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_159 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T435 = div(X_T430, X_T434)
// Elementwise op: [[pid(Add, Switch)]] X_T436 = add(X_T435, X_I_159)
// Elementwise op: X_T437 = cmp_lt(X_T436, X_T2)
// Elementwise op: [[pid(Relu)]] X_T438 = cond(X_T437, X_T2, X_T436)
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
__kernel void kernel_c108_sdk_131(__global float* restrict  X_T438, __global const float* restrict  X_T430, __global const float* restrict  X_T434, __global const float* restrict  X_I_159)
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
      float LX_T430 = X_T430[gout_idx];
      float LX_T434 = X_T434[(i4_gid + i4_tid)];
      float LX_I_159 = X_I_159[(i4_gid + i4_tid)];
      float LX_T435 = (LX_T430 / LX_T434);
      float LX_T436 = (LX_T435 + LX_I_159);
      int LX_T437 = (LX_T436 < 0.0f);
      float LX_T438 = select((float)LX_T436, (float)0.0f, (int)LX_T437);
      X_T438[gout_idx] = LX_T438;
    }
  }
}
