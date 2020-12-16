#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 31 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 992 }
// Out stride: { 194432, 13888, 992, 1 }
// Elementwise input X_T1158 shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Elementwise input X_T1162 shape: fp32(992):(1):3.875 KiB
// Elementwise input X_I_440 shape: fp32(992):(1):3.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1163 = div(X_T1158, X_T1162)
// Elementwise op: [[pid(Add, Switch)]] X_T1164 = add(X_T1163, X_I_440)
// Elementwise op: X_T1165 = cmp_lt(X_T1164, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1166 = cond(X_T1165, X_T2, X_T1164)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 992):(194432, 13888, 992, 1):759.5 KiB
// Computed true ops: 777728
// Computed work groups: 217
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 31, 1
__kernel void kernel_c124_sdk_389(__global float* restrict  X_T1166, __global const float* restrict  X_T1158, __global const float* restrict  X_T1162, __global const float* restrict  X_I_440)
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
      float LX_T1158 = X_T1158[gout_idx];
      float LX_T1162 = X_T1162[(i4_gid + i4_tid)];
      float LX_I_440 = X_I_440[(i4_gid + i4_tid)];
      float LX_T1163 = (LX_T1158 / LX_T1162);
      float LX_T1164 = (LX_T1163 + LX_I_440);
      int LX_T1165 = (LX_T1164 < 0.0f);
      float LX_T1166 = select((float)LX_T1164, (float)0.0f, (int)LX_T1165);
      X_T1166[gout_idx] = LX_T1166;
    }
  }
}
