#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1632 }
// Out stride: { 79968, 11424, 1632, 1 }
// Elementwise input X_T2351 shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Elementwise input X_T2374 shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Elementwise input X_I_923 shape: fp32(1632):(1):6.375 KiB
// Elementwise input X_I_922 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2375 = add(X_T2351, X_T2374)
// Elementwise op: [[pid(Sub)]] X_T2377 = sub(X_T2375, X_I_923)
// Elementwise op: [[pid(Mul)]] X_T2378 = mul(X_T2377, X_I_922)
// Tile size: { 1, 1, 1, 1632 }
// Contraction output var shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Computed true ops: 239904
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 816
// Computed mem write: 13056
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_824(__global float* restrict  X_T2375, __global float* restrict  X_T2378, __global const float* restrict  X_T2351, __global const float* restrict  X_T2374, __global const float* restrict  X_I_923, __global const float* restrict  X_I_922)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11424 * i2_gid) + (1632 * i3_gid)) + i4);
      float LX_T2351 = X_T2351[gout_idx];
      float LX_T2374 = X_T2374[gout_idx];
      float LX_I_923 = X_I_923[i4];
      float LX_I_922 = X_I_922[i4];
      float LX_T2375 = (LX_T2351 + LX_T2374);
      float LX_T2377 = (LX_T2375 - LX_I_923);
      float LX_T2378 = (LX_T2377 * LX_I_922);
      X_T2375[gout_idx] = LX_T2375;
      X_T2378[gout_idx] = LX_T2378;
    }
  }
}
