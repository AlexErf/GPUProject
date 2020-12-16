#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_114 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T104 = add(X_T76, X_I_114)
// Elementwise op: X_T105 = cmp_lt(X_T104, X_T6)
// Elementwise op: [[pid(Sqrt)]] X_T106 = cond(X_T105, X_T6, X_T104)
// Elementwise op: [[pid(Sqrt)]] X_T107 = sqrt(X_T106)
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
__kernel void kernel_c25_sdk_23(__global float* restrict  X_T107, __global const float* restrict  X_I_114)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_114 = X_I_114[i1_tid];
  float LX_T104 = (0.0010000000474974513f + LX_I_114);
  int LX_T105 = (LX_T104 < (float)0);
  float LX_T106 = select((float)LX_T104, (float)0, (int)LX_T105);
  float LX_T107 = native_sqrt(LX_T106);
  X_T107[i1_tid] = LX_T107;
}
