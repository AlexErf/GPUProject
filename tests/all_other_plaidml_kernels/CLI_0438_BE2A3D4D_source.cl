#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 31 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 992 }
// Out stride: { 194432, 13888, 992, 1 }
// Elementwise input X_T1131 shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Elementwise input X_T1154 shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Elementwise input X_I_442 shape: fp32(992):(1):3.875 KiB
// Elementwise input X_I_441 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1155 = add(X_T1131, X_T1154)
// Elementwise op: [[pid(Sub)]] X_T1157 = sub(X_T1155, X_I_442)
// Elementwise op: [[pid(Mul)]] X_T1158 = mul(X_T1157, X_I_441)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Computed true ops: 583296
// Computed work groups: 217
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 31, 1
__kernel void kernel_c124_sdk_386(__global float* restrict  X_T1155, __global float* restrict  X_T1158, __global const float* restrict  X_T1131, __global const float* restrict  X_T1154, __global const float* restrict  X_I_442, __global const float* restrict  X_I_441)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((13888 * (i2_gid + i2_tid)) + (992 * i3)) + (i4_gid + i4_tid));
      float LX_T1131 = X_T1131[gout_idx];
      float LX_T1154 = X_T1154[gout_idx];
      float LX_I_442 = X_I_442[(i4_gid + i4_tid)];
      float LX_I_441 = X_I_441[(i4_gid + i4_tid)];
      float LX_T1155 = (LX_T1131 + LX_T1154);
      float LX_T1157 = (LX_T1155 - LX_I_442);
      float LX_T1158 = (LX_T1157 * LX_I_441);
      X_T1155[gout_idx] = LX_T1155;
      X_T1158[gout_idx] = LX_T1158;
    }
  }
}
