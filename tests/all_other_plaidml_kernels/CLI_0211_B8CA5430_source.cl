#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 128 }
// Out stride: { 100352, 3584, 128, 1 }
// Elementwise input X_T255 shape: fp32(1, 28, 28, 128):(100352, 3584, 128, 1):392 KiB
// Elementwise input X_T259 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_89 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T260 = div(X_T255, X_T259)
// Elementwise op: [[pid(Add, Switch)]] X_T261 = add(X_T260, X_I_89)
// Elementwise op: X_T262 = cmp_lt(X_T261, X_T2)
// Elementwise op: [[pid(Relu)]] X_T263 = cond(X_T262, X_T2, X_T261)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 128):(100352, 3584, 128, 1):392 KiB
// Computed true ops: 401408
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
// gwork = 3584, 2, 1
__kernel void kernel_c108_sdk_68(__global float* restrict  X_T263, __global const float* restrict  X_T255, __global const float* restrict  X_T259, __global const float* restrict  X_I_89)
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
      int gout_idx = (((3584 * i2) + (128 * (i3_gid + i3_tid))) + (i4_gid + i4));
      float LX_T255 = X_T255[gout_idx];
      float LX_T259 = X_T259[(i4_gid + i4)];
      float LX_I_89 = X_I_89[(i4_gid + i4)];
      float LX_T260 = (LX_T255 / LX_T259);
      float LX_T261 = (LX_T260 + LX_I_89);
      int LX_T262 = (LX_T261 < 0.0f);
      float LX_T263 = select((float)LX_T261, (float)0.0f, (int)LX_T262);
      X_T263[gout_idx] = LX_T263;
    }
  }
}
