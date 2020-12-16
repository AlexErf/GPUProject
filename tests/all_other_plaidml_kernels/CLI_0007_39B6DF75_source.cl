#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16 1 1
// lid: 16 1 1
// Names: { i1 }
// Ranges: { 16 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(16):(1):64 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 16 }
// Contraction output var shape: fp32(16):(1):64 bytes
// Computed true ops: 16
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 16
// Computed rollups: 0
// Computed threads used: 16
// lwork = 16, 1, 1
// gwork = 16, 1, 1
__kernel void kernel_c5_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 16);
  float LX_I_0 = X_I_0[i1_tid];
  float LX_T0 = LX_I_0;
  X_T0[i1_tid] = LX_T0;
}
