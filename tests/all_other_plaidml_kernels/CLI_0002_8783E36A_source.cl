#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(32):(1):128 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 32 }
// Contraction output var shape: fp32(32):(1):128 bytes
// Computed true ops: 32
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 32
// Computed rollups: 0
// Computed threads used: 32
// lwork = 32, 1, 1
// gwork = 32, 1, 1
__kernel void kernel_c2_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_0 = X_I_0[i1_tid];
  float LX_T0 = LX_I_0;
  X_T0[i1_tid] = LX_T0;
}
