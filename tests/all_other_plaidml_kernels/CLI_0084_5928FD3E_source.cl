#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_33 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Add)]] X_T34 = add(X_T33, X_I_33)
// Elementwise op: X_T35 = cmp_lt(X_T34, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T36 = cond(X_T35, X_T4, X_T34)
// Elementwise op: [[pid(Sqrt)]] X_T37 = sqrt(X_T36)
// Tile size: { 32 }
// Contraction output var shape: fp32(32):(1):128 bytes
// Computed true ops: 128
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 32
// Computed rollups: 0
// Computed threads used: 32
// lwork = 32, 1, 1
// gwork = 32, 1, 1
__kernel void kernel_c51_sdk_1(__global float* restrict  X_T37, __global const float* restrict  X_I_33)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_33 = X_I_33[i1_tid];
  float LX_T34 = (0.0010000000474974513f + LX_I_33);
  int LX_T35 = (LX_T34 < (float)0);
  float LX_T36 = select((float)LX_T34, (float)0, (int)LX_T35);
  float LX_T37 = native_sqrt(LX_T36);
  X_T37[i1_tid] = LX_T37;
}
