#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 416 }
// Out stride: { 326144, 11648, 416, 1 }
// Elementwise input X_T453 shape: fp32(1, 28, 28, 416):(326144, 11648, 416, 1):1274 KiB
// Elementwise input X_T476 shape: fp32(1, 28, 28, 416):(326144, 11648, 416, 1):1274 KiB
// Elementwise input X_I_181 shape: fp32(416):(1):1.625 KiB
// Elementwise input X_I_180 shape: fp32(416):(1):1.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T477 = add(X_T453, X_T476)
// Elementwise op: [[pid(Sub)]] X_T479 = sub(X_T477, X_I_181)
// Elementwise op: [[pid(Mul)]] X_T480 = mul(X_T479, X_I_180)
// Tile size: { 1, 4, 1, 416 }
// Contraction output var shape: fp32(1, 28, 28, 416):(326144, 11648, 416, 1):1274 KiB
// Computed true ops: 978432
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 832
// Computed mem write: 13312
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c108_sdk_146(__global float* restrict  X_T477, __global float* restrict  X_T480, __global const float* restrict  X_T453, __global const float* restrict  X_T476, __global const float* restrict  X_I_181, __global const float* restrict  X_I_180)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((11648 * (i2_gid + i2_tid)) + (416 * i3_gid)) + i4);
      float LX_T453 = X_T453[gout_idx];
      float LX_T476 = X_T476[gout_idx];
      float LX_I_181 = X_I_181[i4];
      float LX_I_180 = X_I_180[i4];
      float LX_T477 = (LX_T453 + LX_T476);
      float LX_T479 = (LX_T477 - LX_I_181);
      float LX_T480 = (LX_T479 * LX_I_180);
      X_T477[gout_idx] = LX_T477;
      X_T480[gout_idx] = LX_T480;
    }
  }
}
