#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 18 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 576 }
// Out stride: { 112896, 8064, 576, 1 }
// Elementwise input X_T798 shape: fp32(1, 14, 14, 576):(112896, 8064, 576, 1):441 KiB
// Elementwise input X_T821 shape: fp32(1, 14, 14, 576):(112896, 8064, 576, 1):441 KiB
// Elementwise input X_I_312 shape: fp32(576):(1):2.25 KiB
// Elementwise input X_I_311 shape: fp32(576):(1):2.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T822 = add(X_T798, X_T821)
// Elementwise op: [[pid(Sub)]] X_T824 = sub(X_T822, X_I_312)
// Elementwise op: [[pid(Mul)]] X_T825 = mul(X_T824, X_I_311)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 576):(112896, 8064, 576, 1):441 KiB
// Computed true ops: 338688
// Computed work groups: 126
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 18, 1
__kernel void kernel_c108_sdk_269(__global float* restrict  X_T822, __global float* restrict  X_T825, __global const float* restrict  X_T798, __global const float* restrict  X_T821, __global const float* restrict  X_I_312, __global const float* restrict  X_I_311)
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
      float LX_T798 = X_T798[gout_idx];
      float LX_T821 = X_T821[gout_idx];
      float LX_I_312 = X_I_312[(i4_gid + i4_tid)];
      float LX_I_311 = X_I_311[(i4_gid + i4_tid)];
      float LX_T822 = (LX_T798 + LX_T821);
      float LX_T824 = (LX_T822 - LX_I_312);
      float LX_T825 = (LX_T824 * LX_I_311);
      X_T822[gout_idx] = LX_T822;
      X_T825[gout_idx] = LX_T825;
    }
  }
}