#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 84 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(84):(1):336 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 84 }
// Contraction output var shape: fp32(84):(1):336 bytes
// Computed true ops: 84
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 84
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c12_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 84);
  if (i1_cond)
  {
    float LX_I_0 = X_I_0[i1_tid];
    float LX_T0 = LX_I_0;
    X_T0[i1_tid] = LX_T0;
  }
}
