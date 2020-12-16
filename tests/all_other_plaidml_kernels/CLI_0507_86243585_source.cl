#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 42 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1344 }
// Out stride: { 263424, 18816, 1344, 1 }
// Elementwise input X_T1433 shape: fp32(1, 14, 14, 1344):(263424, 18816, 1344, 1):1029 KiB
// Elementwise input X_T1437 shape: fp32(1344):(1):5.25 KiB
// Elementwise input X_I_550 shape: fp32(1344):(1):5.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1438 = div(X_T1433, X_T1437)
// Elementwise op: [[pid(Add, Switch)]] X_T1439 = add(X_T1438, X_I_550)
// Elementwise op: X_T1440 = cmp_lt(X_T1439, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1441 = cond(X_T1440, X_T2, X_T1439)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1344):(263424, 18816, 1344, 1):1029 KiB
// Computed true ops: 1053696
// Computed work groups: 294
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 42, 1
__kernel void kernel_c124_sdk_488(__global float* restrict  X_T1441, __global const float* restrict  X_T1433, __global const float* restrict  X_T1437, __global const float* restrict  X_I_550)
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
      int gout_idx = (((18816 * (i2_gid + i2_tid)) + (1344 * i3)) + (i4_gid + i4_tid));
      float LX_T1433 = X_T1433[gout_idx];
      float LX_T1437 = X_T1437[(i4_gid + i4_tid)];
      float LX_I_550 = X_I_550[(i4_gid + i4_tid)];
      float LX_T1438 = (LX_T1433 / LX_T1437);
      float LX_T1439 = (LX_T1438 + LX_I_550);
      int LX_T1440 = (LX_T1439 < 0.0f);
      float LX_T1441 = select((float)LX_T1439, (float)0.0f, (int)LX_T1440);
      X_T1441[gout_idx] = LX_T1441;
    }
  }
}
