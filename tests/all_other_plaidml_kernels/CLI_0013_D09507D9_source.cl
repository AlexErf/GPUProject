#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i2 }
// Ranges: { 80 }
// Out stride: { 1 }
// Elementwise input X_I_12 shape: fp32(1, 80):(80, 1):320 bytes
// Elementwise op: [[pid(Cast)]] X_T24 = as_int(X_I_12, X_T23)
// Tile size: { 80 }
// Contraction output var shape: i32(1, 80):(80, 1):320 bytes
// Computed true ops: 80
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 80
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_9(__global int* restrict  X_T24, __global const float* restrict  X_I_12)
{
  int tid = get_local_id(0);
  int i2_tid = (tid % 128);
  int i2_cond = (i2_tid < 80);
  if (i2_cond)
  {
    float LX_I_12 = X_I_12[i2_tid];
    int LX_T24 = (int)LX_I_12;
    X_T24[i2_tid] = LX_T24;
  }
}
