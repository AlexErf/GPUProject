#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1632 }
// Out stride: { 79968, 11424, 1632, 1 }
// Elementwise input X_T2143 shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Elementwise input X_T2166 shape: fp32(1, 7, 7, 1632):(79968, 11424, 1632, 1):312.375 KiB
// Elementwise input X_I_843 shape: fp32(1632):(1):6.375 KiB
// Elementwise input X_I_842 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2167 = add(X_T2143, X_T2166)
// Elementwise op: [[pid(Sub)]] X_T2169 = sub(X_T2167, X_I_843)
// Elementwise op: [[pid(Mul)]] X_T2170 = mul(X_T2169, X_I_842)
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
__kernel void kernel_c108_sdk_752(__global float* restrict  X_T2167, __global float* restrict  X_T2170, __global const float* restrict  X_T2143, __global const float* restrict  X_T2166, __global const float* restrict  X_I_843, __global const float* restrict  X_I_842)
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
      float LX_T2143 = X_T2143[gout_idx];
      float LX_T2166 = X_T2166[gout_idx];
      float LX_I_843 = X_I_843[i4];
      float LX_I_842 = X_I_842[i4];
      float LX_T2167 = (LX_T2143 + LX_T2166);
      float LX_T2169 = (LX_T2167 - LX_I_843);
      float LX_T2170 = (LX_T2169 * LX_I_842);
      X_T2167[gout_idx] = LX_T2167;
      X_T2170[gout_idx] = LX_T2170;
    }
  }
}
