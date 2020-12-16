#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1600 }
// Out stride: { 78400, 11200, 1600, 1 }
// Elementwise input X_T2118 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_T2141 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_I_833 shape: fp32(1600):(1):6.25 KiB
// Elementwise input X_I_832 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2142 = add(X_T2118, X_T2141)
// Elementwise op: [[pid(Sub)]] X_T2144 = sub(X_T2142, X_I_833)
// Elementwise op: [[pid(Mul)]] X_T2145 = mul(X_T2144, X_I_832)
// Tile size: { 1, 1, 1, 1600 }
// Contraction output var shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Computed true ops: 235200
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 800
// Computed mem write: 12800
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_743(__global float* restrict  X_T2142, __global float* restrict  X_T2145, __global const float* restrict  X_T2118, __global const float* restrict  X_T2141, __global const float* restrict  X_I_833, __global const float* restrict  X_I_832)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11200 * i2_gid) + (1600 * i3_gid)) + i4);
      float LX_T2118 = X_T2118[gout_idx];
      float LX_T2141 = X_T2141[gout_idx];
      float LX_I_833 = X_I_833[i4];
      float LX_I_832 = X_I_832[i4];
      float LX_T2142 = (LX_T2118 + LX_T2141);
      float LX_T2144 = (LX_T2142 - LX_I_833);
      float LX_T2145 = (LX_T2144 * LX_I_832);
      X_T2142[gout_idx] = LX_T2142;
      X_T2145[gout_idx] = LX_T2145;
    }
  }
}
