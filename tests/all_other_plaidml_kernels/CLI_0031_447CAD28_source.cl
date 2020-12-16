#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_248 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T38 = add(X_T37, X_I_248)
// Elementwise op: X_T39 = cmp_lt(X_T38, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T40 = cond(X_T39, X_T3, X_T38)
// Elementwise op: [[pid(Sqrt)]] X_T41 = sqrt(X_T40)
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
__kernel void kernel_c29_sdk_2(__global float* restrict  X_T41, __global const float* restrict  X_I_248)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_248 = X_I_248[i1_tid];
  float LX_T38 = (0.0010000000474974513f + LX_I_248);
  int LX_T39 = (LX_T38 < (float)0);
  float LX_T40 = select((float)LX_T38, (float)0, (int)LX_T39);
  float LX_T41 = native_sqrt(LX_T40);
  X_T41[i1_tid] = LX_T41;
}
