#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(160):(1):640 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 160 }
// Contraction output var shape: fp32(160):(1):640 bytes
// Computed true ops: 160
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 20
// Computed mem write: 640
// Computed operations: 160
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c32_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_0 = X_I_0[i1_tid];
    float LX_T0 = LX_I_0;
    X_T0[i1_tid] = LX_T0;
  }
}
