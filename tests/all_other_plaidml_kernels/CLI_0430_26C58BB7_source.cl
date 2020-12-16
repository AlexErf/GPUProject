#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 896 }
// Out stride: { 43904, 6272, 896, 1 }
// Elementwise input X_T1475 shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Elementwise input X_T1479 shape: fp32(896):(1):3.5 KiB
// Elementwise input X_I_571 shape: fp32(896):(1):3.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1480 = div(X_T1475, X_T1479)
// Elementwise op: [[pid(Add, Switch)]] X_T1481 = add(X_T1480, X_I_571)
// Elementwise op: X_T1482 = cmp_lt(X_T1481, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1483 = cond(X_T1482, X_T2, X_T1481)
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
__kernel void kernel_c68_sdk_512(__global float* restrict  X_T1483, __global const float* restrict  X_T1475, __global const float* restrict  X_T1479, __global const float* restrict  X_I_571)
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
      float LX_T1475 = X_T1475[gout_idx];
      float LX_T1479 = X_T1479[i4];
      float LX_I_571 = X_I_571[i4];
      float LX_T1480 = (LX_T1475 / LX_T1479);
      float LX_T1481 = (LX_T1480 + LX_I_571);
      int LX_T1482 = (LX_T1481 < 0.0f);
      float LX_T1483 = select((float)LX_T1481, (float)0.0f, (int)LX_T1482);
      X_T1483[gout_idx] = LX_T1483;
    }
  }
}
