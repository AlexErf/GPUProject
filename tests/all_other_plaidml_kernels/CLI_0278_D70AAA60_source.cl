#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 51968 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 51744 }
// Out stride: { 1 }
// Elementwise input X_T2514 shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Elementwise input X_T2542 shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2543 = add(X_T2514, X_T2542)
// Elementwise op: X_T2544 = cmp_lt(X_T2543, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2545 = cond(X_T2544, X_T1, X_T2543)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Computed true ops: 155232
// Computed work groups: 203
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 51968, 1, 1
__kernel void kernel_c42_sdk_978(__global float* restrict  X_T2545, __global const float* restrict  X_T2514, __global const float* restrict  X_T2542)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 51712) || (i2_i3_i4_tid < 32));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T2514 = X_T2514[gout_idx];
    float LX_T2542 = X_T2542[gout_idx];
    float LX_T2543 = (LX_T2514 + LX_T2542);
    int LX_T2544 = (LX_T2543 < 0.0f);
    float LX_T2545 = select((float)LX_T2543, (float)0.0f, (int)LX_T2544);
    X_T2545[gout_idx] = LX_T2545;
  }
}
