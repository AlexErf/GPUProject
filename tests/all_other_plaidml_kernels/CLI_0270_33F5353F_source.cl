#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 544 }
// Out stride: { 106624, 7616, 544, 1 }
// Elementwise input X_T753 shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Elementwise input X_T776 shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Elementwise input X_I_302 shape: fp32(544):(1):2.125 KiB
// Elementwise input X_I_301 shape: fp32(544):(1):2.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T777 = add(X_T753, X_T776)
// Elementwise op: [[pid(Sub)]] X_T779 = sub(X_T777, X_I_302)
// Elementwise op: [[pid(Mul)]] X_T780 = mul(X_T779, X_I_301)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Computed true ops: 319872
// Computed work groups: 119
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 17, 1
__kernel void kernel_c68_sdk_260(__global float* restrict  X_T777, __global float* restrict  X_T780, __global const float* restrict  X_T753, __global const float* restrict  X_T776, __global const float* restrict  X_I_302, __global const float* restrict  X_I_301)
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
      int gout_idx = (((7616 * (i2_gid + i2_tid)) + (544 * i3)) + (i4_gid + i4_tid));
      float LX_T753 = X_T753[gout_idx];
      float LX_T776 = X_T776[gout_idx];
      float LX_I_302 = X_I_302[(i4_gid + i4_tid)];
      float LX_I_301 = X_I_301[(i4_gid + i4_tid)];
      float LX_T777 = (LX_T753 + LX_T776);
      float LX_T779 = (LX_T777 - LX_I_302);
      float LX_T780 = (LX_T779 * LX_I_301);
      X_T777[gout_idx] = LX_T777;
      X_T780[gout_idx] = LX_T780;
    }
  }
}
