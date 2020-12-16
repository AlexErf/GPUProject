#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 448 }
// Out stride: { 351232, 12544, 448, 1 }
// Elementwise input X_T513 shape: fp32(1, 28, 28, 448):(351232, 12544, 448, 1):1372 KiB
// Elementwise input X_T517 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_189 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T518 = div(X_T513, X_T517)
// Elementwise op: [[pid(Add, Switch)]] X_T519 = add(X_T518, X_I_189)
// Elementwise op: X_T520 = cmp_lt(X_T519, X_T2)
// Elementwise op: [[pid(Relu)]] X_T521 = cond(X_T520, X_T2, X_T519)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 448):(351232, 12544, 448, 1):1372 KiB
// Computed true ops: 1404928
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c124_sdk_158(__global float* restrict  X_T521, __global const float* restrict  X_T513, __global const float* restrict  X_T517, __global const float* restrict  X_I_189)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
  {
    int i2 = ((4 * i2_lid) + i2_tid);
    int gout_idx = (((12544 * i2) + (448 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T513 = X_T513[gout_idx];
    float LX_T517 = X_T517[(i4_gid + i4_tid)];
    float LX_I_189 = X_I_189[(i4_gid + i4_tid)];
    float LX_T518 = (LX_T513 / LX_T517);
    float LX_T519 = (LX_T518 + LX_I_189);
    int LX_T520 = (LX_T519 < 0.0f);
    float LX_T521 = select((float)LX_T519, (float)0.0f, (int)LX_T520);
    X_T521[gout_idx] = LX_T521;
  }
}
