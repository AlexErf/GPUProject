#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 896 }
// Out stride: { 43904, 6272, 896, 1 }
// Elementwise input X_T1568 shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Elementwise input X_T1591 shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Elementwise input X_I_613 shape: fp32(896):(1):3.5 KiB
// Elementwise input X_I_612 shape: fp32(896):(1):3.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1592 = add(X_T1568, X_T1591)
// Elementwise op: [[pid(Sub)]] X_T1594 = sub(X_T1592, X_I_613)
// Elementwise op: [[pid(Mul)]] X_T1595 = mul(X_T1594, X_I_612)
// Tile size: { 1, 1, 1, 896 }
// Contraction output var shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Computed true ops: 131712
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_545(__global float* restrict  X_T1592, __global float* restrict  X_T1595, __global const float* restrict  X_T1568, __global const float* restrict  X_T1591, __global const float* restrict  X_I_613, __global const float* restrict  X_I_612)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6272 * i2_gid) + (896 * i3_gid)) + i4);
      float LX_T1568 = X_T1568[gout_idx];
      float LX_T1591 = X_T1591[gout_idx];
      float LX_I_613 = X_I_613[i4];
      float LX_I_612 = X_I_612[i4];
      float LX_T1592 = (LX_T1568 + LX_T1591);
      float LX_T1594 = (LX_T1592 - LX_I_613);
      float LX_T1595 = (LX_T1594 * LX_I_612);
      X_T1592[gout_idx] = LX_T1592;
      X_T1595[gout_idx] = LX_T1595;
    }
  }
}
