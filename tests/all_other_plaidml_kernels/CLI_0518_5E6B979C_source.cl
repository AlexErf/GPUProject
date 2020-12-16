#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 896 }
// Out stride: { 43904, 6272, 896, 1 }
// Elementwise input X_T1595 shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Elementwise input X_T1599 shape: fp32(896):(1):3.5 KiB
// Elementwise input X_I_611 shape: fp32(896):(1):3.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1600 = div(X_T1595, X_T1599)
// Elementwise op: [[pid(Add, Switch)]] X_T1601 = add(X_T1600, X_I_611)
// Elementwise op: X_T1602 = cmp_lt(X_T1601, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1603 = cond(X_T1602, X_T2, X_T1601)
// Tile size: { 1, 1, 1, 896 }
// Contraction output var shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Computed true ops: 175616
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_548(__global float* restrict  X_T1603, __global const float* restrict  X_T1595, __global const float* restrict  X_T1599, __global const float* restrict  X_I_611)
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
      float LX_T1595 = X_T1595[gout_idx];
      float LX_T1599 = X_T1599[i4];
      float LX_I_611 = X_I_611[i4];
      float LX_T1600 = (LX_T1595 / LX_T1599);
      float LX_T1601 = (LX_T1600 + LX_I_611);
      int LX_T1602 = (LX_T1601 < 0.0f);
      float LX_T1603 = select((float)LX_T1601, (float)0.0f, (int)LX_T1602);
      X_T1603[gout_idx] = LX_T1603;
    }
  }
}
