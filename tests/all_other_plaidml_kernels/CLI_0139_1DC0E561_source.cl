#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 34560 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 68992 }
// Out stride: { 1 }
// Elementwise input X_T426 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise input X_T456 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T457 = add(X_T426, X_T456)
// Elementwise op: X_T458 = cmp_lt(X_T457, X_T1)
// Elementwise op: [[pid(Relu)]] X_T459 = cond(X_T458, X_T1, X_T457)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Computed true ops: 206976
// Computed work groups: 135
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 34560, 1, 1
__kernel void kernel_c42_sdk_159(__global float* restrict  X_T459, __global const float* restrict  X_T426, __global const float* restrict  X_T456)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 68608) || (i2_i3_i4_tid < 128)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T426 = X_T426[gout_idx];
      float LX_T456 = X_T456[gout_idx];
      float LX_T457 = (LX_T426 + LX_T456);
      int LX_T458 = (LX_T457 < 0.0f);
      float LX_T459 = select((float)LX_T457, (float)0.0f, (int)LX_T458);
      X_T459[gout_idx] = LX_T459;
    }
  }
}
