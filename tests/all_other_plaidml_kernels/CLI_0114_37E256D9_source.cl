#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 22 }
// Out stride: { 68992, 1232, 22, 1 }
// Elementwise input X_T279 shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Elementwise input X_T283 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_120 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T284 = div(X_T279, X_T283)
// Elementwise op: [[pid(Add, Switch)]] X_T285 = add(X_T284, X_I_120)
// Elementwise op: X_T286 = cmp_lt(X_T285, X_T1)
// Elementwise op: [[pid(Relu)]] X_T287 = cond(X_T286, X_T1, X_T285)
// Tile size: { 1, 56, 2, 22 }
// Contraction output var shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Computed true ops: 275968
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_95(__global float* restrict  X_T287, __global const float* restrict  X_T279, __global const float* restrict  X_T283, __global const float* restrict  X_I_120)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = (i4_tid < 22);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1232 * i2) + (22 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T279 = X_T279[gout_idx];
      float LX_T283 = X_T283[i4_tid];
      float LX_I_120 = X_I_120[i4_tid];
      float LX_T284 = (LX_T279 / LX_T283);
      float LX_T285 = (LX_T284 + LX_I_120);
      int LX_T286 = (LX_T285 < 0.0f);
      float LX_T287 = select((float)LX_T285, (float)0.0f, (int)LX_T286);
      X_T287[gout_idx] = LX_T287;
    }
  }
}
