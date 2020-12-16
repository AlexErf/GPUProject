#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 960 }
// Out stride: { 47040, 6720, 960, 1 }
// Elementwise input X_T608 shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Elementwise input X_I_235 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_234 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(Sub)]] X_T609 = sub(X_T608, X_I_235)
// Elementwise op: [[pid(Mul)]] X_T610 = mul(X_T609, X_I_234)
// Tile size: { 1, 1, 1, 960 }
// Contraction output var shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Computed true ops: 94080
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 360
// Computed mem write: 3840
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_165(__global float* restrict  X_T610, __global const float* restrict  X_T608, __global const float* restrict  X_I_235, __global const float* restrict  X_I_234)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6720 * i2_gid) + (960 * i3_gid)) + i4);
      float LX_T608 = X_T608[gout_idx];
      float LX_I_235 = X_I_235[i4];
      float LX_I_234 = X_I_234[i4];
      float LX_T609 = (LX_T608 - LX_I_235);
      float LX_T610 = (LX_T609 * LX_I_234);
      X_T610[gout_idx] = LX_T610;
    }
  }
}
