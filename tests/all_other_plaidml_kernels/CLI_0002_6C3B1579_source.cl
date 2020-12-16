#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4096 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 4096 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(1, 1, 64, 64):(4096, 4096, 64, 1):16 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 1, 64, 64):(4096, 4096, 64, 1):16 KiB
// Computed true ops: 4096
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4096, 1, 1
__kernel void kernel_c3_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 256);
  int i3_i4_tid = (tid % 256);
  int gout_idx = (i3_i4_gid + i3_i4_tid);
  float LX_I_0 = X_I_0[gout_idx];
  float LX_T0 = LX_I_0;
  X_T0[gout_idx] = LX_T0;
}
