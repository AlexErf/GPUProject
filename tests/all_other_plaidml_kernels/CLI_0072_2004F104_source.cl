#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_91 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Add)]] X_T60 = add(X_T59, X_I_91)
// Elementwise op: X_T61 = cmp_lt(X_T60, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T62 = cond(X_T61, X_T4, X_T60)
// Elementwise op: [[pid(Sqrt)]] X_T63 = sqrt(X_T62)
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
__kernel void kernel_c43_sdk_9(__global float* restrict  X_T63, __global const float* restrict  X_I_91)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_91 = X_I_91[i1_tid];
  float LX_T60 = (0.0010000000474974513f + LX_I_91);
  int LX_T61 = (LX_T60 < (float)0);
  float LX_T62 = select((float)LX_T60, (float)0, (int)LX_T61);
  float LX_T63 = native_sqrt(LX_T62);
  X_T63[i1_tid] = LX_T63;
}
