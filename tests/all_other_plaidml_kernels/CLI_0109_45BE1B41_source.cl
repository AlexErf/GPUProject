#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 98048 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 392000 }
// Out stride: { 1 }
// Elementwise input X_T95 shape: fp32(1, 35, 35, 320):(392000, 11200, 320, 1):1531.25 KiB
// Elementwise input X_T121 shape: fp32(1, 35, 35, 320):(392000, 11200, 320, 1):1531.25 KiB
// Elementwise op: X_T122 = add(X_T95, X_T121)
// Tile size: { 1024 }
// Contraction output var shape: fp32(1, 35, 35, 320):(392000, 11200, 320, 1):1531.25 KiB
// Computed true ops: 392000
// Computed work groups: 383
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 98048, 1, 1
__kernel void kernel_c51_sdk_28(__global float* restrict  X_T122, __global const float* restrict  X_T95, __global const float* restrict  X_T121)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 1024);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 4; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 3) || ((i2_i3_i4_gid != 391168) || (i2_i3_i4_tid < 64)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T95 = X_T95[gout_idx];
      float LX_T121 = X_T121[gout_idx];
      float LX_T122 = (LX_T95 + LX_T121);
      X_T122[gout_idx] = LX_T122;
    }
  }
}
