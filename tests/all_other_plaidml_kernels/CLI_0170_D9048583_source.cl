#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 889088 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 1778112 }
// Out stride: { 1 }
// Elementwise input X_T658 shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Elementwise input X_T686 shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T687 = add(X_T658, X_T686)
// Elementwise op: X_T688 = cmp_lt(X_T687, X_T1)
// Elementwise op: [[pid(Relu)]] X_T689 = cond(X_T688, X_T1, X_T687)
// Tile size: { 512 }
// Contraction output var shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Computed true ops: 5334336
// Computed work groups: 3473
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 889088, 1, 1
__kernel void kernel_c42_sdk_250(__global float* restrict  X_T689, __global const float* restrict  X_T658, __global const float* restrict  X_T686)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 512);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 2; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4_cond = ((i2_i3_i4_lid < 1) || ((i2_i3_i4_gid != 1777664) || (i2_i3_i4_tid < 192)));
    if (i2_i3_i4_cond)
    {
      int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
      int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
      float LX_T658 = X_T658[gout_idx];
      float LX_T686 = X_T686[gout_idx];
      float LX_T687 = (LX_T658 + LX_T686);
      int LX_T688 = (LX_T687 < 0.0f);
      float LX_T689 = select((float)LX_T687, (float)0.0f, (int)LX_T688);
      X_T689[gout_idx] = LX_T689;
    }
  }
}
