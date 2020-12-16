#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 800 }
// Out stride: { 39200, 5600, 800, 1 }
// Elementwise input X_T1400 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_T1404 shape: fp32(800):(1):3.125 KiB
// Elementwise input X_I_541 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1405 = div(X_T1400, X_T1404)
// Elementwise op: [[pid(Add, Switch)]] X_T1406 = add(X_T1405, X_I_541)
// Elementwise op: X_T1407 = cmp_lt(X_T1406, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1408 = cond(X_T1407, X_T2, X_T1406)
// Tile size: { 1, 1, 1, 800 }
// Contraction output var shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Computed true ops: 156800
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 300
// Computed mem write: 3200
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_485(__global float* restrict  X_T1408, __global const float* restrict  X_T1400, __global const float* restrict  X_T1404, __global const float* restrict  X_I_541)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((5600 * i2_gid) + (800 * i3_gid)) + i4);
      float LX_T1400 = X_T1400[gout_idx];
      float LX_T1404 = X_T1404[i4];
      float LX_I_541 = X_I_541[i4];
      float LX_T1405 = (LX_T1400 / LX_T1404);
      float LX_T1406 = (LX_T1405 + LX_I_541);
      int LX_T1407 = (LX_T1406 < 0.0f);
      float LX_T1408 = select((float)LX_T1406, (float)0.0f, (int)LX_T1407);
      X_T1408[gout_idx] = LX_T1408;
    }
  }
}
