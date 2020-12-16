#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 39 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1248 }
// Out stride: { 244608, 17472, 1248, 1 }
// Elementwise input X_T1350 shape: fp32(1, 14, 14, 1248):(244608, 17472, 1248, 1):955.5 KiB
// Elementwise input X_T1354 shape: fp32(1248):(1):4.875 KiB
// Elementwise input X_I_520 shape: fp32(1248):(1):4.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1355 = div(X_T1350, X_T1354)
// Elementwise op: [[pid(Add, Switch)]] X_T1356 = add(X_T1355, X_I_520)
// Elementwise op: X_T1357 = cmp_lt(X_T1356, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1358 = cond(X_T1357, X_T2, X_T1356)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1248):(244608, 17472, 1248, 1):955.5 KiB
// Computed true ops: 978432
// Computed work groups: 273
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 39, 1
__kernel void kernel_c108_sdk_461(__global float* restrict  X_T1358, __global const float* restrict  X_T1350, __global const float* restrict  X_T1354, __global const float* restrict  X_I_520)
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
      int gout_idx = (((17472 * (i2_gid + i2_tid)) + (1248 * i3)) + (i4_gid + i4_tid));
      float LX_T1350 = X_T1350[gout_idx];
      float LX_T1354 = X_T1354[(i4_gid + i4_tid)];
      float LX_I_520 = X_I_520[(i4_gid + i4_tid)];
      float LX_T1355 = (LX_T1350 / LX_T1354);
      float LX_T1356 = (LX_T1355 + LX_I_520);
      int LX_T1357 = (LX_T1356 < 0.0f);
      float LX_T1358 = select((float)LX_T1356, (float)0.0f, (int)LX_T1357);
      X_T1358[gout_idx] = LX_T1358;
    }
  }
}
