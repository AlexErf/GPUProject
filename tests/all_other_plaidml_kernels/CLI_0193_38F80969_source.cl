#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 384 }
// Out stride: { 301056, 10752, 384, 1 }
// Elementwise input X_T408 shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Elementwise input X_T431 shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Elementwise input X_I_171 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_170 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T432 = add(X_T408, X_T431)
// Elementwise op: [[pid(Sub)]] X_T434 = sub(X_T432, X_I_171)
// Elementwise op: [[pid(Mul)]] X_T435 = mul(X_T434, X_I_170)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Computed true ops: 903168
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c68_sdk_137(__global float* restrict  X_T432, __global float* restrict  X_T435, __global const float* restrict  X_T408, __global const float* restrict  X_T431, __global const float* restrict  X_I_171, __global const float* restrict  X_I_170)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((10752 * i2) + (384 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T408 = X_T408[gout_idx];
      float LX_T431 = X_T431[gout_idx];
      float LX_I_171 = X_I_171[(i4_gid + i4)];
      float LX_I_170 = X_I_170[(i4_gid + i4)];
      float LX_T432 = (LX_T408 + LX_T431);
      float LX_T434 = (LX_T432 - LX_I_171);
      float LX_T435 = (LX_T434 * LX_I_170);
      X_T432[gout_idx] = LX_T432;
      X_T435[gout_idx] = LX_T435;
    }
  }
}
