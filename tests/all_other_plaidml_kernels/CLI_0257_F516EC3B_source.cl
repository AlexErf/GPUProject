#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 325376 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 325248 }
// Out stride: { 1 }
// Elementwise input X_T2983 shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Elementwise input X_T3013 shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T3014 = add(X_T2983, X_T3013)
// Elementwise op: X_T3015 = cmp_lt(X_T3014, X_T1)
// Elementwise op: [[pid(Relu)]] X_T3016 = cond(X_T3015, X_T1, X_T3014)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 11, 11, 2688):(325248, 29568, 2688, 1):1270.5 KiB
// Computed true ops: 975744
// Computed work groups: 1271
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 325376, 1, 1
__kernel void kernel_c42_sdk_1165(__global float* restrict  X_T3016, __global const float* restrict  X_T2983, __global const float* restrict  X_T3013)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 325120) || (i2_i3_i4_tid < 128));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T2983 = X_T2983[gout_idx];
    float LX_T3013 = X_T3013[gout_idx];
    float LX_T3014 = (LX_T2983 + LX_T3013);
    int LX_T3015 = (LX_T3014 < 0.0f);
    float LX_T3016 = select((float)LX_T3014, (float)0.0f, (int)LX_T3015);
    X_T3016[gout_idx] = LX_T3016;
  }
}
