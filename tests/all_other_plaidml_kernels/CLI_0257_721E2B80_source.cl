#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 448 }
// Out stride: { 87808, 6272, 448, 1 }
// Elementwise input X_T705 shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Elementwise input X_T709 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_270 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T710 = div(X_T705, X_T709)
// Elementwise op: [[pid(Add, Switch)]] X_T711 = add(X_T710, X_I_270)
// Elementwise op: X_T712 = cmp_lt(X_T711, X_T2)
// Elementwise op: [[pid(Relu)]] X_T713 = cond(X_T712, X_T2, X_T711)
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
__kernel void kernel_c68_sdk_236(__global float* restrict  X_T713, __global const float* restrict  X_T705, __global const float* restrict  X_T709, __global const float* restrict  X_I_270)
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
    float LX_T705 = X_T705[gout_idx];
    float LX_T709 = X_T709[i4];
    float LX_I_270 = X_I_270[i4];
    float LX_T710 = (LX_T705 / LX_T709);
    float LX_T711 = (LX_T710 + LX_I_270);
    int LX_T712 = (LX_T711 < 0.0f);
    float LX_T713 = select((float)LX_T711, (float)0.0f, (int)LX_T712);
    X_T713[gout_idx] = LX_T713;
  }
}
