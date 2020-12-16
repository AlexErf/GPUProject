#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1664 }
// Out stride: { 81536, 11648, 1664, 1 }
// Elementwise input X_T2168 shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Elementwise input X_T2191 shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Elementwise input X_I_4 shape: fp32(1664):(1):6.5 KiB
// Elementwise input X_I_3 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2192 = add(X_T2168, X_T2191)
// Elementwise op: [[pid(Sub)]] X_T2193 = sub(X_T2192, X_I_4)
// Elementwise op: [[pid(Mul)]] X_T2194 = mul(X_T2193, X_I_3)
// Tile size: { 1, 1, 1, 1664 }
// Contraction output var shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Computed true ops: 244608
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 832
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_761(__global float* restrict  X_T2194, __global const float* restrict  X_T2168, __global const float* restrict  X_T2191, __global const float* restrict  X_I_4, __global const float* restrict  X_I_3)
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
      float LX_T2168 = X_T2168[gout_idx];
      float LX_T2191 = X_T2191[gout_idx];
      float LX_I_4 = X_I_4[i4];
      float LX_I_3 = X_I_3[i4];
      float LX_T2192 = (LX_T2168 + LX_T2191);
      float LX_T2193 = (LX_T2192 - LX_I_4);
      float LX_T2194 = (LX_T2193 * LX_I_3);
      X_T2194[gout_idx] = LX_T2194;
    }
  }
}
