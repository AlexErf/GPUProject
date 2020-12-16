#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 24 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 768 }
// Out stride: { 150528, 10752, 768, 1 }
// Elementwise input X_T983 shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Elementwise input X_T987 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_370 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T988 = div(X_T983, X_T987)
// Elementwise op: [[pid(Add, Switch)]] X_T989 = add(X_T988, X_I_370)
// Elementwise op: X_T990 = cmp_lt(X_T989, X_T2)
// Elementwise op: [[pid(Relu)]] X_T991 = cond(X_T990, X_T2, X_T989)
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
__kernel void kernel_c124_sdk_326(__global float* restrict  X_T991, __global const float* restrict  X_T983, __global const float* restrict  X_T987, __global const float* restrict  X_I_370)
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
      float LX_T983 = X_T983[gout_idx];
      float LX_T987 = X_T987[(i4_gid + i4_tid)];
      float LX_I_370 = X_I_370[(i4_gid + i4_tid)];
      float LX_T988 = (LX_T983 / LX_T987);
      float LX_T989 = (LX_T988 + LX_I_370);
      int LX_T990 = (LX_T989 < 0.0f);
      float LX_T991 = select((float)LX_T989, (float)0.0f, (int)LX_T990);
      X_T991[gout_idx] = LX_T991;
    }
  }
}
