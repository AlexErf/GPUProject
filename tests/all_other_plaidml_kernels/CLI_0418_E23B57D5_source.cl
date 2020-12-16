#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 832 }
// Out stride: { 40768, 5824, 832, 1 }
// Elementwise input X_T1398 shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Elementwise input X_T1421 shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Elementwise input X_I_553 shape: fp32(832):(1):3.25 KiB
// Elementwise input X_I_552 shape: fp32(832):(1):3.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1422 = add(X_T1398, X_T1421)
// Elementwise op: [[pid(Sub)]] X_T1424 = sub(X_T1422, X_I_553)
// Elementwise op: [[pid(Mul)]] X_T1425 = mul(X_T1424, X_I_552)
// Tile size: { 1, 1, 1, 832 }
// Contraction output var shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Computed true ops: 122304
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 416
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_491(__global float* restrict  X_T1422, __global float* restrict  X_T1425, __global const float* restrict  X_T1398, __global const float* restrict  X_T1421, __global const float* restrict  X_I_553, __global const float* restrict  X_I_552)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((5824 * i2_gid) + (832 * i3_gid)) + i4);
      float LX_T1398 = X_T1398[gout_idx];
      float LX_T1421 = X_T1421[gout_idx];
      float LX_I_553 = X_I_553[i4];
      float LX_I_552 = X_I_552[i4];
      float LX_T1422 = (LX_T1398 + LX_T1421);
      float LX_T1424 = (LX_T1422 - LX_I_553);
      float LX_T1425 = (LX_T1424 * LX_I_552);
      X_T1422[gout_idx] = LX_T1422;
      X_T1425[gout_idx] = LX_T1425;
    }
  }
}
