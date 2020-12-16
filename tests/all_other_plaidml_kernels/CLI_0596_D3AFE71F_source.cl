#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 896 }
// Out stride: { 43904, 6272, 896, 1 }
// Elementwise input X_T1803 shape: fp32(1, 7, 7, 896):(43904, 6272, 896, 1):171.5 KiB
// Elementwise input X_T1807 shape: fp32(896):(1):3.5 KiB
// Elementwise input X_I_691 shape: fp32(896):(1):3.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1808 = div(X_T1803, X_T1807)
// Elementwise op: [[pid(Add, Switch)]] X_T1809 = add(X_T1808, X_I_691)
// Elementwise op: X_T1810 = cmp_lt(X_T1809, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1811 = cond(X_T1810, X_T2, X_T1809)
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
__kernel void kernel_c124_sdk_620(__global float* restrict  X_T1811, __global const float* restrict  X_T1803, __global const float* restrict  X_T1807, __global const float* restrict  X_I_691)
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
      float LX_T1803 = X_T1803[gout_idx];
      float LX_T1807 = X_T1807[i4];
      float LX_I_691 = X_I_691[i4];
      float LX_T1808 = (LX_T1803 / LX_T1807);
      float LX_T1809 = (LX_T1808 + LX_I_691);
      int LX_T1810 = (LX_T1809 < 0.0f);
      float LX_T1811 = select((float)LX_T1809, (float)0.0f, (int)LX_T1810);
      X_T1811[gout_idx] = LX_T1811;
    }
  }
}
