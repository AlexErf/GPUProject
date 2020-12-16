#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1568 }
// Out stride: { 76832, 10976, 1568, 1 }
// Elementwise input X_T2093 shape: fp32(1, 7, 7, 1568):(76832, 10976, 1568, 1):300.125 KiB
// Elementwise input X_T2116 shape: fp32(1, 7, 7, 1568):(76832, 10976, 1568, 1):300.125 KiB
// Elementwise input X_I_823 shape: fp32(1568):(1):6.125 KiB
// Elementwise input X_I_822 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2117 = add(X_T2093, X_T2116)
// Elementwise op: [[pid(Sub)]] X_T2119 = sub(X_T2117, X_I_823)
// Elementwise op: [[pid(Mul)]] X_T2120 = mul(X_T2119, X_I_822)
// Tile size: { 1, 1, 1, 1568 }
// Contraction output var shape: fp32(1, 7, 7, 1568):(76832, 10976, 1568, 1):300.125 KiB
// Computed true ops: 230496
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 784
// Computed mem write: 12544
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_734(__global float* restrict  X_T2117, __global float* restrict  X_T2120, __global const float* restrict  X_T2093, __global const float* restrict  X_T2116, __global const float* restrict  X_I_823, __global const float* restrict  X_I_822)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((10976 * i2_gid) + (1568 * i3_gid)) + i4);
      float LX_T2093 = X_T2093[gout_idx];
      float LX_T2116 = X_T2116[gout_idx];
      float LX_I_823 = X_I_823[i4];
      float LX_I_822 = X_I_822[i4];
      float LX_T2117 = (LX_T2093 + LX_T2116);
      float LX_T2119 = (LX_T2117 - LX_I_823);
      float LX_T2120 = (LX_T2119 * LX_I_822);
      X_T2117[gout_idx] = LX_T2117;
      X_T2120[gout_idx] = LX_T2120;
    }
  }
}
