#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 192 }
// Out stride: { 150528, 5376, 192, 1 }
// Elementwise input X_T313 shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Elementwise input X_T317 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_109 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T318 = div(X_T313, X_T317)
// Elementwise op: [[pid(Add, Switch)]] X_T319 = add(X_T318, X_I_109)
// Elementwise op: X_T320 = cmp_lt(X_T319, X_T2)
// Elementwise op: [[pid(Relu)]] X_T321 = cond(X_T320, X_T2, X_T319)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 192):(150528, 5376, 192, 1):588 KiB
// Computed true ops: 602112
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
__kernel void kernel_c124_sdk_86(__global float* restrict  X_T321, __global const float* restrict  X_T313, __global const float* restrict  X_T317, __global const float* restrict  X_I_109)
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
    float LX_T313 = X_T313[gout_idx];
    float LX_T317 = X_T317[(i4_gid + i4_tid)];
    float LX_I_109 = X_I_109[(i4_gid + i4_tid)];
    float LX_T318 = (LX_T313 / LX_T317);
    float LX_T319 = (LX_T318 + LX_I_109);
    int LX_T320 = (LX_T319 < 0.0f);
    float LX_T321 = select((float)LX_T319, (float)0.0f, (int)LX_T320);
    X_T321[gout_idx] = LX_T321;
  }
}
