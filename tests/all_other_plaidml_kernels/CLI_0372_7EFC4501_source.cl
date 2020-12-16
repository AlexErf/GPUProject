#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 20 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 640 }
// Out stride: { 125440, 8960, 640, 1 }
// Elementwise input X_T856 shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Elementwise input X_T879 shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Elementwise input X_I_332 shape: fp32(640):(1):2.5 KiB
// Elementwise input X_I_331 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T880 = add(X_T856, X_T879)
// Elementwise op: [[pid(Sub)]] X_T882 = sub(X_T880, X_I_332)
// Elementwise op: [[pid(Mul)]] X_T883 = mul(X_T882, X_I_331)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Computed true ops: 376320
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 20, 1
__kernel void kernel_c124_sdk_287(__global float* restrict  X_T880, __global float* restrict  X_T883, __global const float* restrict  X_T856, __global const float* restrict  X_T879, __global const float* restrict  X_I_332, __global const float* restrict  X_I_331)
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
      int gout_idx = (((8960 * (i2_gid + i2_tid)) + (640 * i3)) + (i4_gid + i4_tid));
      float LX_T856 = X_T856[gout_idx];
      float LX_T879 = X_T879[gout_idx];
      float LX_I_332 = X_I_332[(i4_gid + i4_tid)];
      float LX_I_331 = X_I_331[(i4_gid + i4_tid)];
      float LX_T880 = (LX_T856 + LX_T879);
      float LX_T882 = (LX_T880 - LX_I_332);
      float LX_T883 = (LX_T882 * LX_I_331);
      X_T880[gout_idx] = LX_T880;
      X_T883[gout_idx] = LX_T883;
    }
  }
}
