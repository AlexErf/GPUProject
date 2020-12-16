#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 11
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 21, 21, 672 }
// Out stride: { 296352, 14112, 672, 1 }
// Elementwise input X_T2843 shape: fp32(1, 21, 21, 672):(296352, 14112, 672, 1):1157.62 KiB
// Elementwise input X_T2847 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_1083 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2848 = div(X_T2843, X_T2847)
// Elementwise op: [[pid(Add, Switch)]] X_T2849 = add(X_T2848, X_I_1083)
// Elementwise op: X_T2850 = cmp_lt(X_T2849, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2851 = cond(X_T2850, X_T1, X_T2849)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 21, 21, 672):(296352, 14112, 672, 1):1157.62 KiB
// Computed true ops: 1185408
// Computed work groups: 1331
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 11
__kernel void kernel_c42_sdk_1101(__global float* restrict  X_T2851, __global const float* restrict  X_T2843, __global const float* restrict  X_T2847, __global const float* restrict  X_I_1083)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i4_cond = ((i4_gid != 640) || (i4_tid < 32));
  if (i4_cond)
  {
    int i3_cond = ((i3_gid != 20) || (i3_tid < 1));
    if (i3_cond)
    {
      int i2_cond = ((i2_gid != 20) || (i2_tid < 1));
      if (i2_cond)
      {
        int gout_idx = (((14112 * (i2_gid + i2_tid)) + (672 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
        float LX_T2843 = X_T2843[gout_idx];
        float LX_T2847 = X_T2847[(i4_gid + i4_tid)];
        float LX_I_1083 = X_I_1083[(i4_gid + i4_tid)];
        float LX_T2848 = (LX_T2843 / LX_T2847);
        float LX_T2849 = (LX_T2848 + LX_I_1083);
        int LX_T2850 = (LX_T2849 < 0.0f);
        float LX_T2851 = select((float)LX_T2849, (float)0.0f, (int)LX_T2850);
        X_T2851[gout_idx] = LX_T2851;
      }
    }
  }
}
