#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 26 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 832 }
// Out stride: { 163072, 11648, 832, 1 }
// Elementwise input X_T1005 shape: fp32(1, 14, 14, 832):(163072, 11648, 832, 1):637 KiB
// Elementwise input X_T1009 shape: fp32(832):(1):3.25 KiB
// Elementwise input X_I_390 shape: fp32(832):(1):3.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1010 = div(X_T1005, X_T1009)
// Elementwise op: [[pid(Add, Switch)]] X_T1011 = add(X_T1010, X_I_390)
// Elementwise op: X_T1012 = cmp_lt(X_T1011, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1013 = cond(X_T1012, X_T2, X_T1011)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 832):(163072, 11648, 832, 1):637 KiB
// Computed true ops: 652288
// Computed work groups: 182
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 26, 1
__kernel void kernel_c68_sdk_344(__global float* restrict  X_T1013, __global const float* restrict  X_T1005, __global const float* restrict  X_T1009, __global const float* restrict  X_I_390)
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
      int gout_idx = (((11648 * (i2_gid + i2_tid)) + (832 * i3)) + (i4_gid + i4_tid));
      float LX_T1005 = X_T1005[gout_idx];
      float LX_T1009 = X_T1009[(i4_gid + i4_tid)];
      float LX_I_390 = X_I_390[(i4_gid + i4_tid)];
      float LX_T1010 = (LX_T1005 / LX_T1009);
      float LX_T1011 = (LX_T1010 + LX_I_390);
      int LX_T1012 = (LX_T1011 < 0.0f);
      float LX_T1013 = select((float)LX_T1011, (float)0.0f, (int)LX_T1012);
      X_T1013[gout_idx] = LX_T1013;
    }
  }
}
