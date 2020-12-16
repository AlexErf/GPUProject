#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 384 }
// Out stride: { 301056, 10752, 384, 1 }
// Elementwise input X_T435 shape: fp32(1, 28, 28, 384):(301056, 10752, 384, 1):1176 KiB
// Elementwise input X_T439 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_169 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T440 = div(X_T435, X_T439)
// Elementwise op: [[pid(Add, Switch)]] X_T441 = add(X_T440, X_I_169)
// Elementwise op: X_T442 = cmp_lt(X_T441, X_T2)
// Elementwise op: [[pid(Relu)]] X_T443 = cond(X_T442, X_T2, X_T441)
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
__kernel void kernel_c68_sdk_140(__global float* restrict  X_T443, __global const float* restrict  X_T435, __global const float* restrict  X_T439, __global const float* restrict  X_I_169)
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
      float LX_T435 = X_T435[gout_idx];
      float LX_T439 = X_T439[(i4_gid + i4)];
      float LX_I_169 = X_I_169[(i4_gid + i4)];
      float LX_T440 = (LX_T435 / LX_T439);
      float LX_T441 = (LX_T440 + LX_I_169);
      int LX_T442 = (LX_T441 < 0.0f);
      float LX_T443 = select((float)LX_T441, (float)0.0f, (int)LX_T442);
      X_T443[gout_idx] = LX_T443;
    }
  }
}
