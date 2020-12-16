#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 192 }
// Out stride: { 150528, 5376, 192, 1 }
// Elementwise input X_T218 shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Elementwise input X_T222 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_123 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T223 = div(X_T218, X_T222)
// Elementwise op: [[pid(Add, Switch)]] X_T224 = add(X_T223, X_I_123)
// Elementwise op: X_T225 = cmp_lt(X_T224, X_T3)
// Elementwise op: [[pid(Relu)]] X_T226 = cond(X_T225, X_T3, X_T224)
// Elementwise op: X_T227 = cmp_lt(X_T226, X_T2)
// Elementwise op: [[pid(Relu)]] X_T228 = cond(X_T227, X_T226, X_T2)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Computed true ops: 903168
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c43_sdk_54(__global float* restrict  X_T228, __global const float* restrict  X_T218, __global const float* restrict  X_T222, __global const float* restrict  X_I_123)
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
    int gout_idx = (((5376 * i2) + (192 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T218 = X_T218[gout_idx];
    float LX_T222 = X_T222[(i4_gid + i4_tid)];
    float LX_I_123 = X_I_123[(i4_gid + i4_tid)];
    float LX_T223 = (LX_T218 / LX_T222);
    float LX_T224 = (LX_T223 + LX_I_123);
    int LX_T225 = (LX_T224 < 0.0f);
    float LX_T226 = select((float)LX_T224, (float)0.0f, (int)LX_T225);
    int LX_T227 = (LX_T226 < 6.0f);
    float LX_T228 = select((float)6.0f, (float)LX_T226, (int)LX_T227);
    X_T228[gout_idx] = LX_T228;
  }
}
