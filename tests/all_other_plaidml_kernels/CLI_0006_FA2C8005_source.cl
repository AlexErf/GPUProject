#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 512 }
// Out stride: { 1 }
// Elementwise input X_T1 shape: fp32(512):(1):2 KiB
// Elementwise input X_T2 shape: fp32(512):(1):2 KiB
// Elementwise op: X_T3 = add(X_T1, X_T2)
// Tile size: { 256 }
// Contraction output var shape: fp32(512):(1):2 KiB
// Computed true ops: 512
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c3_sdk_2(__global float* restrict  X_T3, __global const float* restrict  X_T1, __global const float* restrict  X_T2)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_T1 = X_T1[gout_idx];
  float LX_T2 = X_T2[gout_idx];
  float LX_T3 = (LX_T1 + LX_T2);
  X_T3[gout_idx] = LX_T3;
}
