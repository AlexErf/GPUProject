#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 512 }
// Out stride: { 401408, 14336, 512, 1 }
// Elementwise input X_T534 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise input X_T538 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_10 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T539 = div(X_T534, X_T538)
// Elementwise op: [[pid(Add, Switch)]] X_T540 = add(X_T539, X_I_10)
// Elementwise op: X_T541 = cmp_lt(X_T540, X_T2)
// Elementwise op: [[pid(Relu)]] X_T542 = cond(X_T541, X_T2, X_T540)
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
__kernel void kernel_c68_sdk_175(__global float* restrict  X_T542, __global const float* restrict  X_T534, __global const float* restrict  X_T538, __global const float* restrict  X_I_10)
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
      float LX_T534 = X_T534[gout_idx];
      float LX_T538 = X_T538[(i4_gid + i4)];
      float LX_I_10 = X_I_10[(i4_gid + i4)];
      float LX_T539 = (LX_T534 / LX_T538);
      float LX_T540 = (LX_T539 + LX_I_10);
      int LX_T541 = (LX_T540 < 0.0f);
      float LX_T542 = select((float)LX_T540, (float)0.0f, (int)LX_T541);
      X_T542[gout_idx] = LX_T542;
    }
  }
}
