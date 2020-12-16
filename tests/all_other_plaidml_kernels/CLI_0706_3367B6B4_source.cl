#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1600 }
// Out stride: { 78400, 11200, 1600, 1 }
// Elementwise input X_T2326 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_T2349 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_I_913 shape: fp32(1600):(1):6.25 KiB
// Elementwise input X_I_912 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2350 = add(X_T2326, X_T2349)
// Elementwise op: [[pid(Sub)]] X_T2352 = sub(X_T2350, X_I_913)
// Elementwise op: [[pid(Mul)]] X_T2353 = mul(X_T2352, X_I_912)
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
__kernel void kernel_c124_sdk_815(__global float* restrict  X_T2350, __global float* restrict  X_T2353, __global const float* restrict  X_T2326, __global const float* restrict  X_T2349, __global const float* restrict  X_I_913, __global const float* restrict  X_I_912)
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
      float LX_T2326 = X_T2326[gout_idx];
      float LX_T2349 = X_T2349[gout_idx];
      float LX_I_913 = X_I_913[i4];
      float LX_I_912 = X_I_912[i4];
      float LX_T2350 = (LX_T2326 + LX_T2349);
      float LX_T2352 = (LX_T2350 - LX_I_913);
      float LX_T2353 = (LX_T2352 * LX_I_912);
      X_T2350[gout_idx] = LX_T2350;
      X_T2353[gout_idx] = LX_T2353;
    }
  }
}
