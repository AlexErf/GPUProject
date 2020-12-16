#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1856 }
// Out stride: { 90944, 12992, 1856, 1 }
// Elementwise input X_T2553 shape: fp32(1, 7, 7, 1856):(90944, 12992, 1856, 1):355.25 KiB
// Elementwise input X_T2557 shape: fp32(1856):(1):7.25 KiB
// Elementwise input X_I_991 shape: fp32(1856):(1):7.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2558 = div(X_T2553, X_T2557)
// Elementwise op: [[pid(Add, Switch)]] X_T2559 = add(X_T2558, X_I_991)
// Elementwise op: X_T2560 = cmp_lt(X_T2559, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2561 = cond(X_T2560, X_T2, X_T2559)
// Tile size: { 1, 1, 1, 1856 }
// Contraction output var shape: fp32(1, 7, 7, 1856):(90944, 12992, 1856, 1):355.25 KiB
// Computed true ops: 363776
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 696
// Computed mem write: 7424
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_890(__global float* restrict  X_T2561, __global const float* restrict  X_T2553, __global const float* restrict  X_T2557, __global const float* restrict  X_I_991)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12992 * i2_gid) + (1856 * i3_gid)) + i4);
      float LX_T2553 = X_T2553[gout_idx];
      float LX_T2557 = X_T2557[i4];
      float LX_I_991 = X_I_991[i4];
      float LX_T2558 = (LX_T2553 / LX_T2557);
      float LX_T2559 = (LX_T2558 + LX_I_991);
      int LX_T2560 = (LX_T2559 < 0.0f);
      float LX_T2561 = select((float)LX_T2559, (float)0.0f, (int)LX_T2560);
      X_T2561[gout_idx] = LX_T2561;
    }
  }
}
