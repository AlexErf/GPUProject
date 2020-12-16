#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 147456 1 1
// lid: 256 1 1
// Names: { i1_i2_i3_i4 }
// Ranges: { 2359296 }
// Out stride: { 1 }
// Elementwise input X_I_0 shape: fp32(3, 3, 512, 512):(786432, 262144, 512, 1):9216 KiB
// Elementwise op: X_T0 = ident(X_I_0)
// Tile size: { 4096 }
// Contraction output var shape: fp32(3, 3, 512, 512):(786432, 262144, 512, 1):9216 KiB
// Computed true ops: 2359296
// Computed work groups: 576
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 16384
// Computed mem read: 512
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 147456, 1, 1
__kernel void kernel_c22_sdk_0(__global float* restrict  X_T0, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  int i1_i2_i3_i4_gid = (get_group_id(0) * 4096);
  int i1_i2_i3_i4_tid = (tid % 256);
  for (int i1_i2_i3_i4_lid = 0; i1_i2_i3_i4_lid < 16; i1_i2_i3_i4_lid += 1)
  {
    int i1_i2_i3_i4 = ((256 * i1_i2_i3_i4_lid) + i1_i2_i3_i4_tid);
    int gout_idx = (i1_i2_i3_i4_gid + i1_i2_i3_i4);
    float LX_I_0 = X_I_0[gout_idx];
    float LX_T0 = LX_I_0;
    X_T0[gout_idx] = LX_T0;
  }
}
