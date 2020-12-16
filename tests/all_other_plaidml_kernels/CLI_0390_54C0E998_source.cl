#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 23 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 736 }
// Out stride: { 144256, 10304, 736, 1 }
// Elementwise input X_T931 shape: fp32(1, 14, 14, 736):(144256, 10304, 736, 1):563.5 KiB
// Elementwise input X_T954 shape: fp32(1, 14, 14, 736):(144256, 10304, 736, 1):563.5 KiB
// Elementwise input X_I_362 shape: fp32(736):(1):2.875 KiB
// Elementwise input X_I_361 shape: fp32(736):(1):2.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T955 = add(X_T931, X_T954)
// Elementwise op: [[pid(Sub)]] X_T957 = sub(X_T955, X_I_362)
// Elementwise op: [[pid(Mul)]] X_T958 = mul(X_T957, X_I_361)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 736):(144256, 10304, 736, 1):563.5 KiB
// Computed true ops: 432768
// Computed work groups: 161
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 23, 1
__kernel void kernel_c124_sdk_314(__global float* restrict  X_T955, __global float* restrict  X_T958, __global const float* restrict  X_T931, __global const float* restrict  X_T954, __global const float* restrict  X_I_362, __global const float* restrict  X_I_361)
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
      int gout_idx = (((10304 * (i2_gid + i2_tid)) + (736 * i3)) + (i4_gid + i4_tid));
      float LX_T931 = X_T931[gout_idx];
      float LX_T954 = X_T954[gout_idx];
      float LX_I_362 = X_I_362[(i4_gid + i4_tid)];
      float LX_I_361 = X_I_361[(i4_gid + i4_tid)];
      float LX_T955 = (LX_T931 + LX_T954);
      float LX_T957 = (LX_T955 - LX_I_362);
      float LX_T958 = (LX_T957 * LX_I_361);
      X_T955[gout_idx] = LX_T955;
      X_T958[gout_idx] = LX_T958;
    }
  }
}
