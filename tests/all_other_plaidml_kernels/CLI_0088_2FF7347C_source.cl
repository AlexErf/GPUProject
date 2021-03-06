#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 24 }
// Out stride: { 75264, 1344, 24, 1 }
// Elementwise input X_T124 shape: fp32(1, 56, 56, 24):(75264, 1344, 24, 1):294 KiB
// Elementwise input X_T128 shape: fp32(24):(1):96 bytes
// Elementwise input X_I_66 shape: fp32(24):(1):96 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T129 = div(X_T124, X_T128)
// Elementwise op: [[pid(Add, Switch)]] X_T130 = add(X_T129, X_I_66)
// Tile size: { 1, 56, 2, 24 }
// Contraction output var shape: fp32(1, 56, 56, 24):(75264, 1344, 24, 1):294 KiB
// Computed true ops: 150528
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
__kernel void kernel_c43_sdk_28(__global float* restrict  X_T130, __global const float* restrict  X_T124, __global const float* restrict  X_T128, __global const float* restrict  X_I_66)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = (i4_tid < 24);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1344 * i2) + (24 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T124 = X_T124[gout_idx];
      float LX_T128 = X_T128[i4_tid];
      float LX_I_66 = X_I_66[i4_tid];
      float LX_T129 = (LX_T124 / LX_T128);
      float LX_T130 = (LX_T129 + LX_I_66);
      X_T130[gout_idx] = LX_T130;
    }
  }
}
