#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 992 }
// Out stride: { 48608, 6944, 992, 1 }
// Elementwise input X_T1670 shape: fp32(1, 7, 7, 992):(48608, 6944, 992, 1):189.875 KiB
// Elementwise input X_T1674 shape: fp32(992):(1):3.875 KiB
// Elementwise input X_I_641 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1675 = div(X_T1670, X_T1674)
// Elementwise op: [[pid(Add, Switch)]] X_T1676 = add(X_T1675, X_I_641)
// Elementwise op: X_T1677 = cmp_lt(X_T1676, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1678 = cond(X_T1677, X_T2, X_T1676)
// Tile size: { 1, 1, 1, 992 }
// Contraction output var shape: fp32(1, 7, 7, 992):(48608, 6944, 992, 1):189.875 KiB
// Computed true ops: 194432
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 372
// Computed mem write: 3968
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_575(__global float* restrict  X_T1678, __global const float* restrict  X_T1670, __global const float* restrict  X_T1674, __global const float* restrict  X_I_641)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 224));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6944 * i2_gid) + (992 * i3_gid)) + i4);
      float LX_T1670 = X_T1670[gout_idx];
      float LX_T1674 = X_T1674[i4];
      float LX_I_641 = X_I_641[i4];
      float LX_T1675 = (LX_T1670 / LX_T1674);
      float LX_T1676 = (LX_T1675 + LX_I_641);
      int LX_T1677 = (LX_T1676 < 0.0f);
      float LX_T1678 = select((float)LX_T1676, (float)0.0f, (int)LX_T1677);
      X_T1678[gout_idx] = LX_T1678;
    }
  }
}
