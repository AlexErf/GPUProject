#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 42 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(42):(1):168 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 42 }
// Contraction output var shape: fp32(42):(1):168 bytes
// Computed true ops: 42
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 42
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c4_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  int i1_cond = (i1_tid < 42);
  if (i1_cond)
  {
    float LX_I_0 = X_I_0[i1_tid];
    float LX_T0 = LX_I_0;
    X_T0[i1_tid] = LX_T0;
  }
}
