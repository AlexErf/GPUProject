#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 256 }
// Out stride: { 200704, 7168, 256, 1 }
// Elementwise input X_T188 shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Elementwise input X_T192 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_75 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T193 = div(X_T188, X_T192)
// Elementwise op: [[pid(Add, Switch)]] X_T194 = add(X_T193, X_I_75)
// Elementwise op: X_T195 = cmp_lt(X_T194, X_T10)
// Elementwise op: [[pid(Relu)]] X_T196 = cond(X_T195, X_T10, X_T194)
// Elementwise op: X_T197 = cmp_lt(X_T196, X_T9)
// Elementwise op: [[pid(Relu)]] X_T198 = cond(X_T197, X_T196, X_T9)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 1204224
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
__kernel void kernel_c25_sdk_47(__global float* restrict  X_T198, __global const float* restrict  X_T188, __global const float* restrict  X_T192, __global const float* restrict  X_I_75)
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
      float LX_T188 = X_T188[gout_idx];
      float LX_T192 = X_T192[(i4_gid + i4)];
      float LX_I_75 = X_I_75[(i4_gid + i4)];
      float LX_T193 = (LX_T188 / LX_T192);
      float LX_T194 = (LX_T193 + LX_I_75);
      int LX_T195 = (LX_T194 < 0.0f);
      float LX_T196 = select((float)LX_T194, (float)0.0f, (int)LX_T195);
      int LX_T197 = (LX_T196 < 6.0f);
      float LX_T198 = select((float)6.0f, (float)LX_T196, (int)LX_T197);
      X_T198[gout_idx] = LX_T198;
    }
  }
}
