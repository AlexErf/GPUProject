#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1184 }
// Out stride: { 232064, 16576, 1184, 1 }
// Elementwise input X_T1308 shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Elementwise input X_T1312 shape: fp32(1184):(1):4.625 KiB
// Elementwise input X_I_500 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1313 = div(X_T1308, X_T1312)
// Elementwise op: [[pid(Add, Switch)]] X_T1314 = add(X_T1313, X_I_500)
// Elementwise op: X_T1315 = cmp_lt(X_T1314, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1316 = cond(X_T1315, X_T2, X_T1314)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Computed true ops: 928256
// Computed work groups: 259
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 37, 1
__kernel void kernel_c124_sdk_443(__global float* restrict  X_T1316, __global const float* restrict  X_T1308, __global const float* restrict  X_T1312, __global const float* restrict  X_I_500)
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
      int gout_idx = (((16576 * (i2_gid + i2_tid)) + (1184 * i3)) + (i4_gid + i4_tid));
      float LX_T1308 = X_T1308[gout_idx];
      float LX_T1312 = X_T1312[(i4_gid + i4_tid)];
      float LX_I_500 = X_I_500[(i4_gid + i4_tid)];
      float LX_T1313 = (LX_T1308 / LX_T1312);
      float LX_T1314 = (LX_T1313 + LX_I_500);
      int LX_T1315 = (LX_T1314 < 0.0f);
      float LX_T1316 = select((float)LX_T1314, (float)0.0f, (int)LX_T1315);
      X_T1316[gout_idx] = LX_T1316;
    }
  }
}
