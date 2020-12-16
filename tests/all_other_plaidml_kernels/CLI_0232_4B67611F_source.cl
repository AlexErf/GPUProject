#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 256 }
// Out stride: { 200704, 7168, 256, 1 }
// Elementwise input X_T355 shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Elementwise input X_T359 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_129 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T360 = div(X_T355, X_T359)
// Elementwise op: [[pid(Add, Switch)]] X_T361 = add(X_T360, X_I_129)
// Elementwise op: X_T362 = cmp_lt(X_T361, X_T2)
// Elementwise op: [[pid(Relu)]] X_T363 = cond(X_T362, X_T2, X_T361)
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
__kernel void kernel_c108_sdk_104(__global float* restrict  X_T363, __global const float* restrict  X_T355, __global const float* restrict  X_T359, __global const float* restrict  X_I_129)
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
      float LX_T355 = X_T355[gout_idx];
      float LX_T359 = X_T359[(i4_gid + i4)];
      float LX_I_129 = X_I_129[(i4_gid + i4)];
      float LX_T360 = (LX_T355 / LX_T359);
      float LX_T361 = (LX_T360 + LX_I_129);
      int LX_T362 = (LX_T361 < 0.0f);
      float LX_T363 = select((float)LX_T361, (float)0.0f, (int)LX_T362);
      X_T363[gout_idx] = LX_T363;
    }
  }
}
