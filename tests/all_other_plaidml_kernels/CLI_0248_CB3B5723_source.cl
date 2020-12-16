#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 11, 11, 672 }
// Out stride: { 81312, 7392, 672, 1 }
// Elementwise input X_T2943 shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Elementwise input X_T2947 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_1102 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2948 = div(X_T2943, X_T2947)
// Elementwise op: [[pid(Add, Switch)]] X_T2949 = add(X_T2948, X_I_1102)
// Tile size: { 1, 4, 1, 64 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 162624
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_1138(__global float* restrict  X_T2949, __global const float* restrict  X_T2943, __global const float* restrict  X_T2947, __global const float* restrict  X_I_1102)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 64);
  int i3_gid = get_group_id(2);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = ((i4_gid != 640) || (i4_tid < 32));
  if (i4_cond)
  {
    int i2_cond = ((i2_gid != 8) || (i2_tid < 3));
    if (i2_cond)
    {
      int gout_idx = (((7392 * (i2_gid + i2_tid)) + (672 * i3_gid)) + (i4_gid + i4_tid));
      float LX_T2943 = X_T2943[gout_idx];
      float LX_T2947 = X_T2947[(i4_gid + i4_tid)];
      float LX_I_1102 = X_I_1102[(i4_gid + i4_tid)];
      float LX_T2948 = (LX_T2943 / LX_T2947);
      float LX_T2949 = (LX_T2948 + LX_I_1102);
      X_T2949[gout_idx] = LX_T2949;
    }
  }
}
