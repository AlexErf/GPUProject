#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 889088 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 1778112 }
// Out stride: { 1 }
// Elementwise input X_T509 shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Elementwise input X_T580 shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Elementwise op: X_T581 = add(X_T509, X_T580)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Computed true ops: 1778112
// Computed work groups: 3473
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 889088, 1, 1
__kernel void kernel_c42_sdk_206(__global float* restrict  X_T581, __global const float* restrict  X_T509, __global const float* restrict  X_T580)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 1777664) || (i2_i3_i4_tid < 192)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T509 = X_T509[gout_idx];
      float LX_T580 = X_T580[gout_idx];
      float LX_T581 = (LX_T509 + LX_T580);
      X_T581[gout_idx] = LX_T581;
    }
  }
}
