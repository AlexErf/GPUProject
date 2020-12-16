#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 288 }
// Out stride: { 225792, 8064, 288, 1 }
// Elementwise input X_T380 shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Elementwise input X_T384 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_139 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T385 = div(X_T380, X_T384)
// Elementwise op: [[pid(Add, Switch)]] X_T386 = add(X_T385, X_I_139)
// Elementwise op: X_T387 = cmp_lt(X_T386, X_T2)
// Elementwise op: [[pid(Relu)]] X_T388 = cond(X_T387, X_T2, X_T386)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Computed true ops: 903168
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c108_sdk_113(__global float* restrict  X_T388, __global const float* restrict  X_T380, __global const float* restrict  X_T384, __global const float* restrict  X_I_139)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 7; i3_lid += 1)
  {
    int i3 = ((4 * i3_lid) + i3_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((8064 * (i2_gid + i2)) + (288 * i3)) + (i4_gid + i4_tid));
      float LX_T380 = X_T380[gout_idx];
      float LX_T384 = X_T384[(i4_gid + i4_tid)];
      float LX_I_139 = X_I_139[(i4_gid + i4_tid)];
      float LX_T385 = (LX_T380 / LX_T384);
      float LX_T386 = (LX_T385 + LX_I_139);
      int LX_T387 = (LX_T386 < 0.0f);
      float LX_T388 = select((float)LX_T386, (float)0.0f, (int)LX_T387);
      X_T388[gout_idx] = LX_T388;
    }
  }
}
