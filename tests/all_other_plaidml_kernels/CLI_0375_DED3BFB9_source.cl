#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 24 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 768 }
// Out stride: { 150528, 10752, 768, 1 }
// Elementwise input X_T975 shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Elementwise input X_T979 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_370 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T980 = div(X_T975, X_T979)
// Elementwise op: [[pid(Add, Switch)]] X_T981 = add(X_T980, X_I_370)
// Elementwise op: X_T982 = cmp_lt(X_T981, X_T2)
// Elementwise op: [[pid(Relu)]] X_T983 = cond(X_T982, X_T2, X_T981)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Computed true ops: 602112
// Computed work groups: 168
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 24, 1
__kernel void kernel_c108_sdk_326(__global float* restrict  X_T983, __global const float* restrict  X_T975, __global const float* restrict  X_T979, __global const float* restrict  X_I_370)
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
      int gout_idx = (((10752 * (i2_gid + i2_tid)) + (768 * i3)) + (i4_gid + i4_tid));
      float LX_T975 = X_T975[gout_idx];
      float LX_T979 = X_T979[(i4_gid + i4_tid)];
      float LX_I_370 = X_I_370[(i4_gid + i4_tid)];
      float LX_T980 = (LX_T975 / LX_T979);
      float LX_T981 = (LX_T980 + LX_I_370);
      int LX_T982 = (LX_T981 < 0.0f);
      float LX_T983 = select((float)LX_T981, (float)0.0f, (int)LX_T982);
      X_T983[gout_idx] = LX_T983;
    }
  }
}
