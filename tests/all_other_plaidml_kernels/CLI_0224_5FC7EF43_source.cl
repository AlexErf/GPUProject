#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 889088 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 889056 }
// Out stride: { 1 }
// Elementwise input X_T1933 shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Elementwise input X_T1961 shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1962 = add(X_T1933, X_T1961)
// Elementwise op: X_T1963 = cmp_lt(X_T1962, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1964 = cond(X_T1963, X_T1, X_T1962)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Computed true ops: 2667168
// Computed work groups: 3473
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 889088, 1, 1
__kernel void kernel_c42_sdk_752(__global float* restrict  X_T1964, __global const float* restrict  X_T1933, __global const float* restrict  X_T1961)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 888832) || (i2_i3_i4_tid < 224));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T1933 = X_T1933[gout_idx];
    float LX_T1961 = X_T1961[gout_idx];
    float LX_T1962 = (LX_T1933 + LX_T1961);
    int LX_T1963 = (LX_T1962 < 0.0f);
    float LX_T1964 = select((float)LX_T1962, (float)0.0f, (int)LX_T1963);
    X_T1964[gout_idx] = LX_T1964;
  }
}
