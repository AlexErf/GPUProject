#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_116 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T133 = add(X_T76, X_I_116)
// Elementwise op: X_T134 = cmp_lt(X_T133, X_T6)
// Elementwise op: [[pid(Sqrt)]] X_T135 = cond(X_T134, X_T6, X_T133)
// Elementwise op: [[pid(Sqrt)]] X_T136 = sqrt(X_T135)
// Tile size: { 128 }
// Contraction output var shape: fp32(128):(1):512 bytes
// Computed true ops: 512
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c25_sdk_31(__global float* restrict  X_T136, __global const float* restrict  X_I_116)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_116 = X_I_116[i1_tid];
  float LX_T133 = (0.0010000000474974513f + LX_I_116);
  int LX_T134 = (LX_T133 < (float)0);
  float LX_T135 = select((float)LX_T133, (float)0, (int)LX_T134);
  float LX_T136 = native_sqrt(LX_T135);
  X_T136[i1_tid] = LX_T136;
}
