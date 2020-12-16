#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 243968 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 487872 }
// Out stride: { 1 }
// Elementwise input X_T3058 shape: fp32(1, 11, 11, 4032):(487872, 44352, 4032, 1):1905.75 KiB
// Elementwise input X_T3129 shape: fp32(1, 11, 11, 4032):(487872, 44352, 4032, 1):1905.75 KiB
// Elementwise op: X_T3130 = add(X_T3058, X_T3129)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 11, 11, 4032):(487872, 44352, 4032, 1):1905.75 KiB
// Computed true ops: 487872
// Computed work groups: 953
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 243968, 1, 1
__kernel void kernel_c42_sdk_1212(__global float* restrict  X_T3130, __global const float* restrict  X_T3058, __global const float* restrict  X_T3129)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 487424) || (i2_i3_i4_tid < 192)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T3058 = X_T3058[gout_idx];
      float LX_T3129 = X_T3129[gout_idx];
      float LX_T3130 = (LX_T3058 + LX_T3129);
      X_T3130[gout_idx] = LX_T3130;
    }
  }
}
