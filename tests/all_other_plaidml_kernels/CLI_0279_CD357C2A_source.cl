#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 512 }
// Out stride: { 401408, 14336, 512, 1 }
// Elementwise input X_T554 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise input X_T558 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_10 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T559 = div(X_T554, X_T558)
// Elementwise op: [[pid(Add, Switch)]] X_T560 = add(X_T559, X_I_10)
// Elementwise op: X_T561 = cmp_lt(X_T560, X_T2)
// Elementwise op: [[pid(Relu)]] X_T562 = cond(X_T561, X_T2, X_T560)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Computed true ops: 1605632
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 8, 1
__kernel void kernel_c108_sdk_175(__global float* restrict  X_T562, __global const float* restrict  X_T554, __global const float* restrict  X_T558, __global const float* restrict  X_I_10)
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
      int gout_idx = (((14336 * i2) + (512 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T554 = X_T554[gout_idx];
      float LX_T558 = X_T558[(i4_gid + i4)];
      float LX_I_10 = X_I_10[(i4_gid + i4)];
      float LX_T559 = (LX_T554 / LX_T558);
      float LX_T560 = (LX_T559 + LX_I_10);
      int LX_T561 = (LX_T560 < 0.0f);
      float LX_T562 = select((float)LX_T560, (float)0.0f, (int)LX_T561);
      X_T562[gout_idx] = LX_T562;
    }
  }
}
