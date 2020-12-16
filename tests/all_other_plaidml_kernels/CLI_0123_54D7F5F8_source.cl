#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 156928 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 156800 }
// Out stride: { 1 }
// Elementwise input X_T187 shape: fp32(1, 35, 35, 128):(156800, 4480, 128, 1):612.5 KiB
// Elementwise input X_T208 shape: fp32(1, 35, 35, 128):(156800, 4480, 128, 1):612.5 KiB
// Elementwise op: X_T209 = add(X_T187, X_T208)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 35, 35, 128):(156800, 4480, 128, 1):612.5 KiB
// Computed true ops: 156800
// Computed work groups: 613
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 156928, 1, 1
__kernel void kernel_c51_sdk_59(__global float* restrict  X_T209, __global const float* restrict  X_T187, __global const float* restrict  X_T208)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 156672) || (i2_i3_i4_tid < 128));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T187 = X_T187[gout_idx];
    float LX_T208 = X_T208[gout_idx];
    float LX_T209 = (LX_T187 + LX_T208);
    X_T209[gout_idx] = LX_T209;
  }
}
