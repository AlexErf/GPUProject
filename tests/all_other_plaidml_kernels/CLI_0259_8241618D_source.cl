#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 288 }
// Out stride: { 225792, 8064, 288, 1 }
// Elementwise input X_T361 shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Elementwise input X_T384 shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Elementwise input X_I_141 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_140 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T385 = add(X_T361, X_T384)
// Elementwise op: [[pid(Sub)]] X_T387 = sub(X_T385, X_I_141)
// Elementwise op: [[pid(Mul)]] X_T388 = mul(X_T387, X_I_140)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Computed true ops: 677376
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c124_sdk_110(__global float* restrict  X_T385, __global float* restrict  X_T388, __global const float* restrict  X_T361, __global const float* restrict  X_T384, __global const float* restrict  X_I_141, __global const float* restrict  X_I_140)
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
      float LX_T361 = X_T361[gout_idx];
      float LX_T384 = X_T384[gout_idx];
      float LX_I_141 = X_I_141[(i4_gid + i4_tid)];
      float LX_I_140 = X_I_140[(i4_gid + i4_tid)];
      float LX_T385 = (LX_T361 + LX_T384);
      float LX_T387 = (LX_T385 - LX_I_141);
      float LX_T388 = (LX_T387 * LX_I_140);
      X_T385[gout_idx] = LX_T385;
      X_T388[gout_idx] = LX_T388;
    }
  }
}
