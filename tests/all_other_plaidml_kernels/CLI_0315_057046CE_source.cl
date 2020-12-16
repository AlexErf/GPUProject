#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 24 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 768 }
// Out stride: { 150528, 10752, 768, 1 }
// Elementwise input X_T955 shape: fp32(1, 14, 14, 768):(150528, 10752, 768, 1):588 KiB
// Elementwise input X_T959 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_370 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T960 = div(X_T955, X_T959)
// Elementwise op: [[pid(Add, Switch)]] X_T961 = add(X_T960, X_I_370)
// Elementwise op: X_T962 = cmp_lt(X_T961, X_T2)
// Elementwise op: [[pid(Relu)]] X_T963 = cond(X_T962, X_T2, X_T961)
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
__kernel void kernel_c68_sdk_326(__global float* restrict  X_T963, __global const float* restrict  X_T955, __global const float* restrict  X_T959, __global const float* restrict  X_I_370)
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
      float LX_T955 = X_T955[gout_idx];
      float LX_T959 = X_T959[(i4_gid + i4_tid)];
      float LX_I_370 = X_I_370[(i4_gid + i4_tid)];
      float LX_T960 = (LX_T955 / LX_T959);
      float LX_T961 = (LX_T960 + LX_I_370);
      int LX_T962 = (LX_T961 < 0.0f);
      float LX_T963 = select((float)LX_T961, (float)0.0f, (int)LX_T962);
      X_T963[gout_idx] = LX_T963;
    }
  }
}
