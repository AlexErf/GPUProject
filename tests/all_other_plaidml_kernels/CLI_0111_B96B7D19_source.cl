#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1184 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 256 }
// Contraction output var shape: fp32(1184):(1):4.625 KiB
// Computed true ops: 1184
// Computed work groups: 5
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 1, 1
__kernel void kernel_c74_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_0 = X_I_0[gout_idx];
    float LX_T0 = LX_I_0;
    X_T0[gout_idx] = LX_T0;
  }
}
