#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1 1 1
// lid: 1 1 1
// Names: {  }
// Ranges: {  }
// Out stride: {  }
// Elementwise input X_I_0 shape: fp32(1):(1):4 bytes
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: {  }
// Contraction output var shape: fp32(1):(1):4 bytes
// Computed true ops: 1
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 1
// Computed rollups: 0
// Computed threads used: 1
// lwork = 1, 1, 1
// gwork = 1, 1, 1
__kernel void kernel_c5_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  float LX_I_0 = X_I_0[0];
  float LX_T0 = LX_I_0;
  X_T0[0] = LX_T0;
}
