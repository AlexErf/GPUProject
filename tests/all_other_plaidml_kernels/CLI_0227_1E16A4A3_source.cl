#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 256 }
// Out stride: { 802816, 14336, 256, 1 }
// Elementwise input X_T215 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_T238 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_I_16 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_15 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Concatenate)]] X_T239 = add(X_T215, X_T238)
// Elementwise op: [[pid(Sub)]] X_T240 = sub(X_T239, X_I_16)
// Elementwise op: [[pid(Mul)]] X_T241 = mul(X_T240, X_I_15)
// Tile size: { 1, 4, 1, 256 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Computed true ops: 2408448
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
__kernel void kernel_c124_sdk_59(__global float* restrict  X_T241, __global const float* restrict  X_T215, __global const float* restrict  X_T238, __global const float* restrict  X_I_16, __global const float* restrict  X_I_15)
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
    float LX_T215 = X_T215[gout_idx];
    float LX_T238 = X_T238[gout_idx];
    float LX_I_16 = X_I_16[i4];
    float LX_I_15 = X_I_15[i4];
    float LX_T239 = (LX_T215 + LX_T238);
    float LX_T240 = (LX_T239 - LX_I_16);
    float LX_T241 = (LX_T240 * LX_I_15);
    X_T241[gout_idx] = LX_T241;
  }
}
