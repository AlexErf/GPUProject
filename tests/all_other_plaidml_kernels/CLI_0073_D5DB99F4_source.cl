#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 256 }
// Out stride: { 50176, 3584, 256, 1 }
// Elementwise input X_T229 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_I_65 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_64 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Sub)]] X_T230 = sub(X_T229, X_I_65)
// Elementwise op: [[pid(Mul)]] X_T231 = mul(X_T230, X_I_64)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 100352
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c25_sdk_57(__global float* restrict  X_T231, __global const float* restrict  X_T229, __global const float* restrict  X_I_65, __global const float* restrict  X_I_64)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((3584 * (i2_gid + i2_tid)) + (256 * i3)) + (i4_gid + i4_tid));
      float LX_T229 = X_T229[gout_idx];
      float LX_I_65 = X_I_65[(i4_gid + i4_tid)];
      float LX_I_64 = X_I_64[(i4_gid + i4_tid)];
      float LX_T230 = (LX_T229 - LX_I_65);
      float LX_T231 = (LX_T230 * LX_I_64);
      X_T231[gout_idx] = LX_T231;
    }
  }
}
