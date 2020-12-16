#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3840 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 928 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(928):(1):3.625 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 64 }
// Contraction output var shape: fp32(928):(1):3.625 KiB
// Computed true ops: 928
// Computed work groups: 15
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 256, 1, 1
// gwork = 3840, 1, 1
__kernel void kernel_c58_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 64);
  int i1_tid = (tid % 64);
  int i1_cond = ((i1_gid != 896) || (i1_tid < 32));
  if (i1_cond)
  {
    if ((tid < 64))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_0 = X_I_0[gout_idx];
      float LX_T0 = LX_I_0;
      X_T0[gout_idx] = LX_T0;
    }
  }
}
