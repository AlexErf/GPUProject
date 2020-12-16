#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 384 }
// Out stride: { 110976, 6528, 384, 1 }
// Elementwise input X_T339 shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Elementwise input X_T343 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_26 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T344 = div(X_T339, X_T343)
// Elementwise op: [[pid(Add, Switch)]] X_T345 = add(X_T344, X_I_26)
// Elementwise op: X_T346 = cmp_lt(X_T345, X_T2)
// Elementwise op: [[pid(Relu)]] X_T347 = cond(X_T346, X_T2, X_T345)
// Tile size: { 1, 1, 1, 384 }
// Contraction output var shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Computed true ops: 443904
// Computed work groups: 289
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 17, 1
__kernel void kernel_c56_sdk_112(__global float* restrict  X_T347, __global const float* restrict  X_T339, __global const float* restrict  X_T343, __global const float* restrict  X_I_26)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6528 * i2_gid) + (384 * i3_gid)) + i4);
      float LX_T339 = X_T339[gout_idx];
      float LX_T343 = X_T343[i4];
      float LX_I_26 = X_I_26[i4];
      float LX_T344 = (LX_T339 / LX_T343);
      float LX_T345 = (LX_T344 + LX_I_26);
      int LX_T346 = (LX_T345 < 0.0f);
      float LX_T347 = select((float)LX_T345, (float)0.0f, (int)LX_T346);
      X_T347[gout_idx] = LX_T347;
    }
  }
}
