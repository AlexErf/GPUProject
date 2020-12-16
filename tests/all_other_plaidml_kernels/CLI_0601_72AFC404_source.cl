#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 928 }
// Out stride: { 45472, 6496, 928, 1 }
// Elementwise input X_T1797 shape: fp32(1, 7, 7, 928):(45472, 6496, 928, 1):177.625 KiB
// Elementwise input X_T1824 shape: fp32(1, 7, 7, 928):(45472, 6496, 928, 1):177.625 KiB
// Elementwise input X_I_703 shape: fp32(928):(1):3.625 KiB
// Elementwise input X_I_702 shape: fp32(928):(1):3.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1825 = add(X_T1797, X_T1824)
// Elementwise op: [[pid(Sub)]] X_T1827 = sub(X_T1825, X_I_703)
// Elementwise op: [[pid(Mul)]] X_T1828 = mul(X_T1827, X_I_702)
// Tile size: { 1, 1, 1, 928 }
// Contraction output var shape: fp32(1, 7, 7, 928):(45472, 6496, 928, 1):177.625 KiB
// Computed true ops: 136416
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 464
// Computed mem write: 7424
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_626(__global float* restrict  X_T1825, __global float* restrict  X_T1828, __global const float* restrict  X_T1797, __global const float* restrict  X_T1824, __global const float* restrict  X_I_703, __global const float* restrict  X_I_702)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 160));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6496 * i2_gid) + (928 * i3_gid)) + i4);
      float LX_T1797 = X_T1797[gout_idx];
      float LX_T1824 = X_T1824[gout_idx];
      float LX_I_703 = X_I_703[i4];
      float LX_I_702 = X_I_702[i4];
      float LX_T1825 = (LX_T1797 + LX_T1824);
      float LX_T1827 = (LX_T1825 - LX_I_703);
      float LX_T1828 = (LX_T1827 * LX_I_702);
      X_T1825[gout_idx] = LX_T1825;
      X_T1828[gout_idx] = LX_T1828;
    }
  }
}
