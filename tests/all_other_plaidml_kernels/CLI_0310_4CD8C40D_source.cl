#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 416 }
// Out stride: { 81536, 5824, 416, 1 }
// Elementwise input X_T673 shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Elementwise input X_T696 shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Elementwise input X_I_262 shape: fp32(416):(1):1.625 KiB
// Elementwise input X_I_261 shape: fp32(416):(1):1.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T697 = add(X_T673, X_T696)
// Elementwise op: [[pid(Sub)]] X_T699 = sub(X_T697, X_I_262)
// Elementwise op: [[pid(Mul)]] X_T700 = mul(X_T699, X_I_261)
// Tile size: { 1, 2, 2, 416 }
// Contraction output var shape: fp32(1, 14, 14, 416):(81536, 5824, 416, 1):318.5 KiB
// Computed true ops: 244608
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 832
// Computed mem write: 13312
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_224(__global float* restrict  X_T697, __global float* restrict  X_T700, __global const float* restrict  X_T673, __global const float* restrict  X_T696, __global const float* restrict  X_I_262, __global const float* restrict  X_I_261)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((5824 * (i2_gid + i2_tid)) + (416 * (i3_gid + i3_tid))) + i4);
      float LX_T673 = X_T673[gout_idx];
      float LX_T696 = X_T696[gout_idx];
      float LX_I_262 = X_I_262[i4];
      float LX_I_261 = X_I_261[i4];
      float LX_T697 = (LX_T673 + LX_T696);
      float LX_T699 = (LX_T697 - LX_I_262);
      float LX_T700 = (LX_T699 * LX_I_261);
      X_T697[gout_idx] = LX_T697;
      X_T700[gout_idx] = LX_T700;
    }
  }
}
