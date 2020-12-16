#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 928 }
// Out stride: { 45472, 6496, 928, 1 }
// Elementwise input X_T1828 shape: fp32(1, 7, 7, 928):(45472, 6496, 928, 1):177.625 KiB
// Elementwise input X_T1832 shape: fp32(928):(1):3.625 KiB
// Elementwise input X_I_701 shape: fp32(928):(1):3.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1833 = div(X_T1828, X_T1832)
// Elementwise op: [[pid(Add, Switch)]] X_T1834 = add(X_T1833, X_I_701)
// Elementwise op: X_T1835 = cmp_lt(X_T1834, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1836 = cond(X_T1835, X_T2, X_T1834)
// Tile size: { 1, 1, 1, 928 }
// Contraction output var shape: fp32(1, 7, 7, 928):(45472, 6496, 928, 1):177.625 KiB
// Computed true ops: 181888
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 348
// Computed mem write: 3712
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_629(__global float* restrict  X_T1836, __global const float* restrict  X_T1828, __global const float* restrict  X_T1832, __global const float* restrict  X_I_701)
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
      float LX_T1828 = X_T1828[gout_idx];
      float LX_T1832 = X_T1832[i4];
      float LX_I_701 = X_I_701[i4];
      float LX_T1833 = (LX_T1828 / LX_T1832);
      float LX_T1834 = (LX_T1833 + LX_I_701);
      int LX_T1835 = (LX_T1834 < 0.0f);
      float LX_T1836 = select((float)LX_T1834, (float)0.0f, (int)LX_T1835);
      X_T1836[gout_idx] = LX_T1836;
    }
  }
}
