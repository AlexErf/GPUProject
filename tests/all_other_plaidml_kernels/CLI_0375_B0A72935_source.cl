#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 20 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 640 }
// Out stride: { 125440, 8960, 640, 1 }
// Elementwise input X_T883 shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Elementwise input X_T887 shape: fp32(640):(1):2.5 KiB
// Elementwise input X_I_330 shape: fp32(640):(1):2.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T888 = div(X_T883, X_T887)
// Elementwise op: [[pid(Add, Switch)]] X_T889 = add(X_T888, X_I_330)
// Elementwise op: X_T890 = cmp_lt(X_T889, X_T2)
// Elementwise op: [[pid(Relu)]] X_T891 = cond(X_T890, X_T2, X_T889)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 640):(125440, 8960, 640, 1):490 KiB
// Computed true ops: 501760
// Computed work groups: 140
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 20, 1
__kernel void kernel_c124_sdk_290(__global float* restrict  X_T891, __global const float* restrict  X_T883, __global const float* restrict  X_T887, __global const float* restrict  X_I_330)
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
      int gout_idx = (((8960 * (i2_gid + i2_tid)) + (640 * i3)) + (i4_gid + i4_tid));
      float LX_T883 = X_T883[gout_idx];
      float LX_T887 = X_T887[(i4_gid + i4_tid)];
      float LX_I_330 = X_I_330[(i4_gid + i4_tid)];
      float LX_T888 = (LX_T883 / LX_T887);
      float LX_T889 = (LX_T888 + LX_I_330);
      int LX_T890 = (LX_T889 < 0.0f);
      float LX_T891 = select((float)LX_T889, (float)0.0f, (int)LX_T890);
      X_T891[gout_idx] = LX_T891;
    }
  }
}
