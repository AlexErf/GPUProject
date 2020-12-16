#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 18 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 576 }
// Out stride: { 112896, 8064, 576, 1 }
// Elementwise input X_T494 shape: fp32(1, 14, 14, 576):(112896, 8064, 576, 1):441 KiB
// Elementwise input X_I_202 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_201 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Sub)]] X_T495 = sub(X_T494, X_I_202)
// Elementwise op: [[pid(Mul)]] X_T496 = mul(X_T495, X_I_201)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 576):(112896, 8064, 576, 1):441 KiB
// Computed true ops: 225792
// Computed work groups: 126
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 18, 1
__kernel void kernel_c43_sdk_132(__global float* restrict  X_T496, __global const float* restrict  X_T494, __global const float* restrict  X_I_202, __global const float* restrict  X_I_201)
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
      int gout_idx = (((8064 * (i2_gid + i2_tid)) + (576 * i3)) + (i4_gid + i4_tid));
      float LX_T494 = X_T494[gout_idx];
      float LX_I_202 = X_I_202[(i4_gid + i4_tid)];
      float LX_I_201 = X_I_201[(i4_gid + i4_tid)];
      float LX_T495 = (LX_T494 - LX_I_202);
      float LX_T496 = (LX_T495 * LX_I_201);
      X_T496[gout_idx] = LX_T496;
    }
  }
}
