#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 800 }
// Out stride: { 39200, 5600, 800, 1 }
// Elementwise input X_T1520 shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Elementwise input X_T1524 shape: fp32(800):(1):3.125 KiB
// Elementwise input X_I_581 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1525 = div(X_T1520, X_T1524)
// Elementwise op: [[pid(Add, Switch)]] X_T1526 = add(X_T1525, X_I_581)
// Elementwise op: X_T1527 = cmp_lt(X_T1526, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1528 = cond(X_T1527, X_T2, X_T1526)
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
__kernel void kernel_c108_sdk_521(__global float* restrict  X_T1528, __global const float* restrict  X_T1520, __global const float* restrict  X_T1524, __global const float* restrict  X_I_581)
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
      float LX_T1520 = X_T1520[gout_idx];
      float LX_T1524 = X_T1524[i4];
      float LX_I_581 = X_I_581[i4];
      float LX_T1525 = (LX_T1520 / LX_T1524);
      float LX_T1526 = (LX_T1525 + LX_I_581);
      int LX_T1527 = (LX_T1526 < 0.0f);
      float LX_T1528 = select((float)LX_T1526, (float)0.0f, (int)LX_T1527);
      X_T1528[gout_idx] = LX_T1528;
    }
  }
}