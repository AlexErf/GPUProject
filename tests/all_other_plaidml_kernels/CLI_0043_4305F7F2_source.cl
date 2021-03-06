#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 256 }
// Out stride: { 802816, 14336, 256, 1 }
// Elementwise input X_T123 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_T127 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_212 shape: fp32(256):(1):1 KiB
// Elementwise input X_T95 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T128 = div(X_T123, X_T127)
// Elementwise op: [[pid(Add, Switch)]] X_T129 = add(X_T128, X_I_212)
// Elementwise op: [[pid(Add)]] X_T130 = add(X_T129, X_T95)
// Elementwise op: X_T131 = cmp_lt(X_T130, X_T2)
// Elementwise op: [[pid(Relu)]] X_T132 = cond(X_T131, X_T2, X_T130)
// Tile size: { 1, 4, 1, 256 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Computed true ops: 4014080
// Computed work groups: 784
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 512
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 56, 1
__kernel void kernel_c29_sdk_26(__global float* restrict  X_T132, __global const float* restrict  X_T123, __global const float* restrict  X_T127, __global const float* restrict  X_I_212, __global const float* restrict  X_T95)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((14336 * (i2_gid + i2_tid)) + (256 * i3_gid)) + i4);
    float LX_T123 = X_T123[gout_idx];
    float LX_T127 = X_T127[i4];
    float LX_I_212 = X_I_212[i4];
    float LX_T95 = X_T95[gout_idx];
    float LX_T128 = (LX_T123 / LX_T127);
    float LX_T129 = (LX_T128 + LX_I_212);
    float LX_T130 = (LX_T129 + LX_T95);
    int LX_T131 = (LX_T130 < 0.0f);
    float LX_T132 = select((float)LX_T130, (float)0.0f, (int)LX_T131);
    X_T132[gout_idx] = LX_T132;
  }
}
