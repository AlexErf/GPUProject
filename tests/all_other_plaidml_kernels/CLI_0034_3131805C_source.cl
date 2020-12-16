#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(64):(1):256 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 64 }
// Contraction output var shape: fp32(64):(1):256 bytes
// Computed true ops: 64
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c21_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_0 = X_I_0[i1_tid];
  float LX_T0 = LX_I_0;
  X_T0[i1_tid] = LX_T0;
}
