#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 34560 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 34496 }
// Out stride: { 1 }
// Elementwise input X_T2292 shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Elementwise input X_T2322 shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2323 = add(X_T2292, X_T2322)
// Elementwise op: X_T2324 = cmp_lt(X_T2323, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2325 = cond(X_T2324, X_T1, X_T2323)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Computed true ops: 103488
// Computed work groups: 135
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 34560, 1, 1
__kernel void kernel_c42_sdk_889(__global float* restrict  X_T2325, __global const float* restrict  X_T2292, __global const float* restrict  X_T2322)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 34304) || (i2_i3_i4_tid < 192));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T2292 = X_T2292[gout_idx];
    float LX_T2322 = X_T2322[gout_idx];
    float LX_T2323 = (LX_T2292 + LX_T2322);
    int LX_T2324 = (LX_T2323 < 0.0f);
    float LX_T2325 = select((float)LX_T2323, (float)0.0f, (int)LX_T2324);
    X_T2325[gout_idx] = LX_T2325;
  }
}
