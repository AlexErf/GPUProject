#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 148224 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 592704 }
// Out stride: { 1 }
// Elementwise input X_T427 shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Elementwise input X_T457 shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T458 = add(X_T427, X_T457)
// Elementwise op: X_T459 = cmp_lt(X_T458, X_T1)
// Elementwise op: [[pid(Relu)]] X_T460 = cond(X_T459, X_T1, X_T458)
// Tile size: { 1024 }
// Contraction output var shape: fp32(1, 42, 42, 336):(592704, 14112, 336, 1):2315.25 KiB
// Computed true ops: 1778112
// Computed work groups: 579
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 148224, 1, 1
__kernel void kernel_c42_sdk_159(__global float* restrict  X_T460, __global const float* restrict  X_T427, __global const float* restrict  X_T457)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 1024);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 4; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 3) || ((i2_i3_i4_gid != 591872) || (i2_i3_i4_tid < 64)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T427 = X_T427[gout_idx];
      float LX_T457 = X_T457[gout_idx];
      float LX_T458 = (LX_T427 + LX_T457);
      int LX_T459 = (LX_T458 < 0.0f);
      float LX_T460 = select((float)LX_T458, (float)0.0f, (int)LX_T459);
      X_T460[gout_idx] = LX_T460;
    }
  }
}
