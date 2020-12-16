#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 256 }
// Out stride: { 200704, 7168, 256, 1 }
// Elementwise input X_T328 shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Elementwise input X_T351 shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Elementwise input X_I_131 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_130 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Concatenate)]] X_T352 = add(X_T328, X_T351)
// Elementwise op: [[pid(Sub)]] X_T354 = sub(X_T352, X_I_131)
// Elementwise op: [[pid(Mul)]] X_T355 = mul(X_T354, X_I_130)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 602112
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 4, 1
__kernel void kernel_c108_sdk_101(__global float* restrict  X_T352, __global float* restrict  X_T355, __global const float* restrict  X_T328, __global const float* restrict  X_T351, __global const float* restrict  X_I_131, __global const float* restrict  X_I_130)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 64);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((7168 * i2) + (256 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T328 = X_T328[gout_idx];
      float LX_T351 = X_T351[gout_idx];
      float LX_I_131 = X_I_131[(i4_gid + i4)];
      float LX_I_130 = X_I_130[(i4_gid + i4)];
      float LX_T352 = (LX_T328 + LX_T351);
      float LX_T354 = (LX_T352 - LX_I_131);
      float LX_T355 = (LX_T354 * LX_I_130);
      X_T352[gout_idx] = LX_T352;
      X_T355[gout_idx] = LX_T355;
    }
  }
}
