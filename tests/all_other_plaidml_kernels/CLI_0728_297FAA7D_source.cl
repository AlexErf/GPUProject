#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1728 }
// Out stride: { 84672, 12096, 1728, 1 }
// Elementwise input X_T2453 shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Elementwise input X_T2457 shape: fp32(1728):(1):6.75 KiB
// Elementwise input X_I_951 shape: fp32(1728):(1):6.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2458 = div(X_T2453, X_T2457)
// Elementwise op: [[pid(Add, Switch)]] X_T2459 = add(X_T2458, X_I_951)
// Elementwise op: X_T2460 = cmp_lt(X_T2459, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2461 = cond(X_T2460, X_T2, X_T2459)
// Tile size: { 1, 1, 1, 1728 }
// Contraction output var shape: fp32(1, 7, 7, 1728):(84672, 12096, 1728, 1):330.75 KiB
// Computed true ops: 338688
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 648
// Computed mem write: 6912
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_854(__global float* restrict  X_T2461, __global const float* restrict  X_T2453, __global const float* restrict  X_T2457, __global const float* restrict  X_I_951)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12096 * i2_gid) + (1728 * i3_gid)) + i4);
      float LX_T2453 = X_T2453[gout_idx];
      float LX_T2457 = X_T2457[i4];
      float LX_I_951 = X_I_951[i4];
      float LX_T2458 = (LX_T2453 / LX_T2457);
      float LX_T2459 = (LX_T2458 + LX_I_951);
      int LX_T2460 = (LX_T2459 < 0.0f);
      float LX_T2461 = select((float)LX_T2459, (float)0.0f, (int)LX_T2460);
      X_T2461[gout_idx] = LX_T2461;
    }
  }
}
