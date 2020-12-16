#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 480 }
// Out stride: { 376320, 13440, 480, 1 }
// Elementwise input X_T538 shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Elementwise input X_T542 shape: fp32(480):(1):1.875 KiB
// Elementwise input X_I_199 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T543 = div(X_T538, X_T542)
// Elementwise op: [[pid(Add, Switch)]] X_T544 = add(X_T543, X_I_199)
// Elementwise op: X_T545 = cmp_lt(X_T544, X_T2)
// Elementwise op: [[pid(Relu)]] X_T546 = cond(X_T545, X_T2, X_T544)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Computed true ops: 1505280
// Computed work groups: 105
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 15, 1
__kernel void kernel_c124_sdk_167(__global float* restrict  X_T546, __global const float* restrict  X_T538, __global const float* restrict  X_T542, __global const float* restrict  X_I_199)
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
      int gout_idx = (((13440 * (i2_gid + i2)) + (480 * i3)) + (i4_gid + i4_tid));
      float LX_T538 = X_T538[gout_idx];
      float LX_T542 = X_T542[(i4_gid + i4_tid)];
      float LX_I_199 = X_I_199[(i4_gid + i4_tid)];
      float LX_T543 = (LX_T538 / LX_T542);
      float LX_T544 = (LX_T543 + LX_I_199);
      int LX_T545 = (LX_T544 < 0.0f);
      float LX_T546 = select((float)LX_T544, (float)0.0f, (int)LX_T545);
      X_T546[gout_idx] = LX_T546;
    }
  }
}
