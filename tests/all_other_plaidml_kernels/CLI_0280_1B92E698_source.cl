#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 384 }
// Out stride: { 301056, 10752, 384, 1 }
// Elementwise input X_T463 shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Elementwise input X_T467 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_169 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T468 = div(X_T463, X_T467)
// Elementwise op: [[pid(Add, Switch)]] X_T469 = add(X_T468, X_I_169)
// Elementwise op: X_T470 = cmp_lt(X_T469, X_T2)
// Elementwise op: [[pid(Relu)]] X_T471 = cond(X_T470, X_T2, X_T469)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Computed true ops: 1204224
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c124_sdk_140(__global float* restrict  X_T471, __global const float* restrict  X_T463, __global const float* restrict  X_T467, __global const float* restrict  X_I_169)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((10752 * i2) + (384 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T463 = X_T463[gout_idx];
      float LX_T467 = X_T467[(i4_gid + i4)];
      float LX_I_169 = X_I_169[(i4_gid + i4)];
      float LX_T468 = (LX_T463 / LX_T467);
      float LX_T469 = (LX_T468 + LX_I_169);
      int LX_T470 = (LX_T469 < 0.0f);
      float LX_T471 = select((float)LX_T469, (float)0.0f, (int)LX_T470);
      X_T471[gout_idx] = LX_T471;
    }
  }
}
