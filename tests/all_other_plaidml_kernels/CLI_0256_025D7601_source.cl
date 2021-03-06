#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 256 }
// Out stride: { 200704, 7168, 256, 1 }
// Elementwise input X_T363 shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Elementwise input X_T367 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_129 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T368 = div(X_T363, X_T367)
// Elementwise op: [[pid(Add, Switch)]] X_T369 = add(X_T368, X_I_129)
// Elementwise op: X_T370 = cmp_lt(X_T369, X_T2)
// Elementwise op: [[pid(Relu)]] X_T371 = cond(X_T370, X_T2, X_T369)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 802816
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 4, 1
__kernel void kernel_c124_sdk_104(__global float* restrict  X_T371, __global const float* restrict  X_T363, __global const float* restrict  X_T367, __global const float* restrict  X_I_129)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 64);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((7168 * i2) + (256 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T363 = X_T363[gout_idx];
      float LX_T367 = X_T367[(i4_gid + i4)];
      float LX_I_129 = X_I_129[(i4_gid + i4)];
      float LX_T368 = (LX_T363 / LX_T367);
      float LX_T369 = (LX_T368 + LX_I_129);
      int LX_T370 = (LX_T369 < 0.0f);
      float LX_T371 = select((float)LX_T369, (float)0.0f, (int)LX_T370);
      X_T371[gout_idx] = LX_T371;
    }
  }
}
