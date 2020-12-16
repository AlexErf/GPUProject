#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 44 }
// Out stride: { 34496, 1232, 44, 1 }
// Elementwise input X_T855 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise input X_T859 shape: fp32(44):(1):176 bytes
// Elementwise input X_I_26 shape: fp32(44):(1):176 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T860 = div(X_T855, X_T859)
// Elementwise op: [[pid(Add, Switch)]] X_T861 = add(X_T860, X_I_26)
// Elementwise op: X_T1076 = cmp_lt(X_T861, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1077 = cond(X_T1076, X_T1, X_T861)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 137984
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c42_sdk_316(__global float* restrict  X_T1077, __global float* restrict  X_T861, __global const float* restrict  X_T855, __global const float* restrict  X_T859, __global const float* restrict  X_I_26)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = ((i4_gid != 32) || (i4_tid < 12));
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1232 * i2) + (44 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
      float LX_T855 = X_T855[gout_idx];
      float LX_T859 = X_T859[(i4_gid + i4_tid)];
      float LX_I_26 = X_I_26[(i4_gid + i4_tid)];
      float LX_T860 = (LX_T855 / LX_T859);
      float LX_T861 = (LX_T860 + LX_I_26);
      int LX_T1076 = (LX_T861 < 0.0f);
      float LX_T1077 = select((float)LX_T861, (float)0.0f, (int)LX_T1076);
      X_T1077[gout_idx] = LX_T1077;
      X_T861[gout_idx] = LX_T861;
    }
  }
}
