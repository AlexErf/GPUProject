#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1664 }
// Out stride: { 81536, 11648, 1664, 1 }
// Elementwise input X_T2376 shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Elementwise input X_T2399 shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Elementwise input X_I_933 shape: fp32(1664):(1):6.5 KiB
// Elementwise input X_I_932 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2400 = add(X_T2376, X_T2399)
// Elementwise op: [[pid(Sub)]] X_T2402 = sub(X_T2400, X_I_933)
// Elementwise op: [[pid(Mul)]] X_T2403 = mul(X_T2402, X_I_932)
// Tile size: { 1, 1, 1, 1664 }
// Contraction output var shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Computed true ops: 244608
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 832
// Computed mem write: 13312
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_833(__global float* restrict  X_T2400, __global float* restrict  X_T2403, __global const float* restrict  X_T2376, __global const float* restrict  X_T2399, __global const float* restrict  X_I_933, __global const float* restrict  X_I_932)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11648 * i2_gid) + (1664 * i3_gid)) + i4);
      float LX_T2376 = X_T2376[gout_idx];
      float LX_T2399 = X_T2399[gout_idx];
      float LX_I_933 = X_I_933[i4];
      float LX_I_932 = X_I_932[i4];
      float LX_T2400 = (LX_T2376 + LX_T2399);
      float LX_T2402 = (LX_T2400 - LX_I_933);
      float LX_T2403 = (LX_T2402 * LX_I_932);
      X_T2400[gout_idx] = LX_T2400;
      X_T2403[gout_idx] = LX_T2403;
    }
  }
}
