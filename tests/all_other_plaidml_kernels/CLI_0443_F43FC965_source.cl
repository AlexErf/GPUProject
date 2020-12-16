#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 992 }
// Out stride: { 48608, 6944, 992, 1 }
// Elementwise input X_T1523 shape: fp32(1, 7, 7, 992):(48608, 6944, 992, 1):189.875 KiB
// Elementwise input X_T1546 shape: fp32(1, 7, 7, 992):(48608, 6944, 992, 1):189.875 KiB
// Elementwise input X_I_603 shape: fp32(992):(1):3.875 KiB
// Elementwise input X_I_602 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1547 = add(X_T1523, X_T1546)
// Elementwise op: [[pid(Sub)]] X_T1549 = sub(X_T1547, X_I_603)
// Elementwise op: [[pid(Mul)]] X_T1550 = mul(X_T1549, X_I_602)
// Tile size: { 1, 1, 1, 992 }
// Contraction output var shape: fp32(1, 7, 7, 992):(48608, 6944, 992, 1):189.875 KiB
// Computed true ops: 145824
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 496
// Computed mem write: 7936
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_536(__global float* restrict  X_T1547, __global float* restrict  X_T1550, __global const float* restrict  X_T1523, __global const float* restrict  X_T1546, __global const float* restrict  X_I_603, __global const float* restrict  X_I_602)
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
      float LX_T1523 = X_T1523[gout_idx];
      float LX_T1546 = X_T1546[gout_idx];
      float LX_I_603 = X_I_603[i4];
      float LX_I_602 = X_I_602[i4];
      float LX_T1547 = (LX_T1523 + LX_T1546);
      float LX_T1549 = (LX_T1547 - LX_I_603);
      float LX_T1550 = (LX_T1549 * LX_I_602);
      X_T1547[gout_idx] = LX_T1547;
      X_T1550[gout_idx] = LX_T1550;
    }
  }
}
