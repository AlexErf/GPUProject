#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 864 }
// Out stride: { 42336, 6048, 864, 1 }
// Elementwise input X_T1450 shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Elementwise input X_T1454 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_561 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1455 = div(X_T1450, X_T1454)
// Elementwise op: [[pid(Add, Switch)]] X_T1456 = add(X_T1455, X_I_561)
// Elementwise op: X_T1457 = cmp_lt(X_T1456, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1458 = cond(X_T1457, X_T2, X_T1456)
// Tile size: { 1, 1, 1, 864 }
// Contraction output var shape: fp32(1, 7, 7, 864):(42336, 6048, 864, 1):165.375 KiB
// Computed true ops: 169344
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 324
// Computed mem write: 3456
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_503(__global float* restrict  X_T1458, __global const float* restrict  X_T1450, __global const float* restrict  X_T1454, __global const float* restrict  X_I_561)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 96));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6048 * i2_gid) + (864 * i3_gid)) + i4);
      float LX_T1450 = X_T1450[gout_idx];
      float LX_T1454 = X_T1454[i4];
      float LX_I_561 = X_I_561[i4];
      float LX_T1455 = (LX_T1450 / LX_T1454);
      float LX_T1456 = (LX_T1455 + LX_I_561);
      int LX_T1457 = (LX_T1456 < 0.0f);
      float LX_T1458 = select((float)LX_T1456, (float)0.0f, (int)LX_T1457);
      X_T1458[gout_idx] = LX_T1458;
    }
  }
}
