#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 44 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1408 }
// Out stride: { 275968, 19712, 1408, 1 }
// Elementwise input X_T1483 shape: fp32(1, 14, 14, 1408):(275968, 19712, 1408, 1):1078 KiB
// Elementwise input X_T1487 shape: fp32(1408):(1):5.5 KiB
// Elementwise input X_I_570 shape: fp32(1408):(1):5.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1488 = div(X_T1483, X_T1487)
// Elementwise op: [[pid(Add, Switch)]] X_T1489 = add(X_T1488, X_I_570)
// Elementwise op: X_T1490 = cmp_lt(X_T1489, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1491 = cond(X_T1490, X_T2, X_T1489)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1408):(275968, 19712, 1408, 1):1078 KiB
// Computed true ops: 1103872
// Computed work groups: 308
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 44, 1
__kernel void kernel_c124_sdk_506(__global float* restrict  X_T1491, __global const float* restrict  X_T1483, __global const float* restrict  X_T1487, __global const float* restrict  X_I_570)
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
      int gout_idx = (((19712 * (i2_gid + i2_tid)) + (1408 * i3)) + (i4_gid + i4_tid));
      float LX_T1483 = X_T1483[gout_idx];
      float LX_T1487 = X_T1487[(i4_gid + i4_tid)];
      float LX_I_570 = X_I_570[(i4_gid + i4_tid)];
      float LX_T1488 = (LX_T1483 / LX_T1487);
      float LX_T1489 = (LX_T1488 + LX_I_570);
      int LX_T1490 = (LX_T1489 < 0.0f);
      float LX_T1491 = select((float)LX_T1489, (float)0.0f, (int)LX_T1490);
      X_T1491[gout_idx] = LX_T1491;
    }
  }
}
