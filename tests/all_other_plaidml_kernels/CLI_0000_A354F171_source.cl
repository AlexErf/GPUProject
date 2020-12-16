#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 9408 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(7, 7, 3, 64):(1344, 192, 64, 1):36.75 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 256 }
// Contraction output var shape: fp32(7, 7, 3, 64):(1344, 192, 64, 1):36.75 KiB
// Computed true ops: 9408
// Computed work groups: 37
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 1, 1
__kernel void kernel_c1_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 256);
  int i1_i2_i3_i4_tid = (tid % 256);
  int i1_i2_i3_i4_cond = ((i1_i2_i3_i4_gid != 9216) || (i1_i2_i3_i4_tid < 192));
  if (i1_i2_i3_i4_cond)
  {
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4_tid);
    float LX_I_0 = X_I_0[gout_idx];
    float LX_T0 = LX_I_0;
    X_T0[gout_idx] = LX_T0;
  }
}
