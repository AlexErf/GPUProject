#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 448 }
// Out stride: { 87808, 6272, 448, 1 }
// Elementwise input X_T725 shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Elementwise input X_T729 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_270 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T730 = div(X_T725, X_T729)
// Elementwise op: [[pid(Add, Switch)]] X_T731 = add(X_T730, X_I_270)
// Elementwise op: X_T732 = cmp_lt(X_T731, X_T2)
// Elementwise op: [[pid(Relu)]] X_T733 = cond(X_T732, X_T2, X_T731)
// Tile size: { 1, 2, 2, 448 }
// Contraction output var shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Computed true ops: 351232
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_236(__global float* restrict  X_T733, __global const float* restrict  X_T725, __global const float* restrict  X_T729, __global const float* restrict  X_I_270)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((6272 * (i2_gid + i2_tid)) + (448 * (i3_gid + i3_tid))) + i4);
    float LX_T725 = X_T725[gout_idx];
    float LX_T729 = X_T729[i4];
    float LX_I_270 = X_I_270[i4];
    float LX_T730 = (LX_T725 / LX_T729);
    float LX_T731 = (LX_T730 + LX_I_270);
    int LX_T732 = (LX_T731 < 0.0f);
    float LX_T733 = select((float)LX_T731, (float)0.0f, (int)LX_T732);
    X_T733[gout_idx] = LX_T733;
  }
}
