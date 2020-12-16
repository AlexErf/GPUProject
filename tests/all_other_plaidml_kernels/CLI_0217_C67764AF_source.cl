#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 512 }
// Out stride: { 401408, 14336, 512, 1 }
// Elementwise input X_T508 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise input X_T531 shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Elementwise input X_I_12 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_11 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Concatenate)]] X_T532 = add(X_T508, X_T531)
// Elementwise op: [[pid(Sub)]] X_T533 = sub(X_T532, X_I_12)
// Elementwise op: [[pid(Mul)]] X_T534 = mul(X_T533, X_I_11)
// Tile size: { 1, 28, 2, 64 }
// Contraction output var shape: fp32(1, 28, 28, 512):(401408, 14336, 512, 1):1568 KiB
// Computed true ops: 1204224
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 8, 1
__kernel void kernel_c68_sdk_173(__global float* restrict  X_T534, __global const float* restrict  X_T508, __global const float* restrict  X_T531, __global const float* restrict  X_I_12, __global const float* restrict  X_I_11)
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
      float LX_T508 = X_T508[gout_idx];
      float LX_T531 = X_T531[gout_idx];
      float LX_I_12 = X_I_12[(i4_gid + i4)];
      float LX_I_11 = X_I_11[(i4_gid + i4)];
      float LX_T532 = (LX_T508 + LX_T531);
      float LX_T533 = (LX_T532 - LX_I_12);
      float LX_T534 = (LX_T533 * LX_I_11);
      X_T534[gout_idx] = LX_T534;
    }
  }
}
