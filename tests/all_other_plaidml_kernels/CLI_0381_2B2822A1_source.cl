#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 21 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 672 }
// Out stride: { 131712, 9408, 672, 1 }
// Elementwise input X_T908 shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Elementwise input X_T912 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_340 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T913 = div(X_T908, X_T912)
// Elementwise op: [[pid(Add, Switch)]] X_T914 = add(X_T913, X_I_340)
// Elementwise op: X_T915 = cmp_lt(X_T914, X_T2)
// Elementwise op: [[pid(Relu)]] X_T916 = cond(X_T915, X_T2, X_T914)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Computed true ops: 526848
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 21, 1
__kernel void kernel_c124_sdk_299(__global float* restrict  X_T916, __global const float* restrict  X_T908, __global const float* restrict  X_T912, __global const float* restrict  X_I_340)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((9408 * (i2_gid + i2_tid)) + (672 * i3)) + (i4_gid + i4_tid));
      float LX_T908 = X_T908[gout_idx];
      float LX_T912 = X_T912[(i4_gid + i4_tid)];
      float LX_I_340 = X_I_340[(i4_gid + i4_tid)];
      float LX_T913 = (LX_T908 / LX_T912);
      float LX_T914 = (LX_T913 + LX_I_340);
      int LX_T915 = (LX_T914 < 0.0f);
      float LX_T916 = select((float)LX_T914, (float)0.0f, (int)LX_T915);
      X_T916[gout_idx] = LX_T916;
    }
  }
}
