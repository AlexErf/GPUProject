#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_35 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T54 = add(X_T33, X_I_35)
// Elementwise op: X_T55 = cmp_lt(X_T54, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T56 = cond(X_T55, X_T4, X_T54)
// Elementwise op: [[pid(Sqrt)]] X_T57 = sqrt(X_T56)
// Tile size: { 64 }
// Contraction output var shape: fp32(64):(1):256 bytes
// Computed true ops: 256
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 64
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c51_sdk_7(__global float* restrict  X_T57, __global const float* restrict  X_I_35)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_35 = X_I_35[i1_tid];
  float LX_T54 = (0.0010000000474974513f + LX_I_35);
  int LX_T55 = (LX_T54 < (float)0);
  float LX_T56 = select((float)LX_T54, (float)0, (int)LX_T55);
  float LX_T57 = native_sqrt(LX_T56);
  X_T57[i1_tid] = LX_T57;
}
