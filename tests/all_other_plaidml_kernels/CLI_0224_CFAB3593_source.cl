#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 103680 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 103488 }
// Out stride: { 1 }
// Elementwise input X_T1587 shape: fp32(1, 14, 14, 528):(103488, 7392, 528, 1):404.25 KiB
// Elementwise input X_T1615 shape: fp32(1, 14, 14, 528):(103488, 7392, 528, 1):404.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1616 = add(X_T1587, X_T1615)
// Elementwise op: X_T1617 = cmp_lt(X_T1616, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1618 = cond(X_T1617, X_T1, X_T1616)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 14, 14, 528):(103488, 7392, 528, 1):404.25 KiB
// Computed true ops: 310464
// Computed work groups: 405
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 103680, 1, 1
__kernel void kernel_c42_sdk_614(__global float* restrict  X_T1618, __global const float* restrict  X_T1587, __global const float* restrict  X_T1615)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 103424) || (i2_i3_i4_tid < 64));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T1587 = X_T1587[gout_idx];
    float LX_T1615 = X_T1615[gout_idx];
    float LX_T1616 = (LX_T1587 + LX_T1615);
    int LX_T1617 = (LX_T1616 < 0.0f);
    float LX_T1618 = select((float)LX_T1616, (float)0.0f, (int)LX_T1617);
    X_T1618[gout_idx] = LX_T1618;
  }
}
