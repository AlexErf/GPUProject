#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 19 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 608 }
// Out stride: { 119168, 8512, 608, 1 }
// Elementwise input X_T850 shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Elementwise input X_T854 shape: fp32(608):(1):2.375 KiB
// Elementwise input X_I_320 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T855 = div(X_T850, X_T854)
// Elementwise op: [[pid(Add, Switch)]] X_T856 = add(X_T855, X_I_320)
// Elementwise op: X_T857 = cmp_lt(X_T856, X_T2)
// Elementwise op: [[pid(Relu)]] X_T858 = cond(X_T857, X_T2, X_T856)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Computed true ops: 476672
// Computed work groups: 133
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 19, 1
__kernel void kernel_c108_sdk_281(__global float* restrict  X_T858, __global const float* restrict  X_T850, __global const float* restrict  X_T854, __global const float* restrict  X_I_320)
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
      int gout_idx = (((8512 * (i2_gid + i2_tid)) + (608 * i3)) + (i4_gid + i4_tid));
      float LX_T850 = X_T850[gout_idx];
      float LX_T854 = X_T854[(i4_gid + i4_tid)];
      float LX_I_320 = X_I_320[(i4_gid + i4_tid)];
      float LX_T855 = (LX_T850 / LX_T854);
      float LX_T856 = (LX_T855 + LX_I_320);
      int LX_T857 = (LX_T856 < 0.0f);
      float LX_T858 = select((float)LX_T856, (float)0.0f, (int)LX_T857);
      X_T858[gout_idx] = LX_T858;
    }
  }
}