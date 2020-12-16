#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5888 1 1
// lid: 256 1 1
// Names: { i3_i4 }
// Ranges: { 11616 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(1, 1, 264, 44):(11616, 11616, 44, 1):45.375 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 1, 264, 44):(11616, 11616, 44, 1):45.375 KiB
// Computed true ops: 11616
// Computed work groups: 23
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 64
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5888, 1, 1
__kernel void kernel_c23_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i3_i4_gid = (get_group_id(0) * 512);
  int i3_i4_tid = (tid % 256);
  for (int i3_i4_lid = 0; i3_i4_lid < 2; i3_i4_lid += 1)
  {
    int i3_i4_cond = ((i3_i4_lid < 1) || ((i3_i4_gid != 11264) || (i3_i4_tid < 96)));
    if (i3_i4_cond)
    {
      int i3_i4 = ((256 * i3_i4_lid) + i3_i4_tid);
      int gout_idx = (i3_i4_gid + i3_i4);
      float LX_I_0 = X_I_0[gout_idx];
      float LX_T0 = LX_I_0;
      X_T0[gout_idx] = LX_T0;
    }
  }
}
