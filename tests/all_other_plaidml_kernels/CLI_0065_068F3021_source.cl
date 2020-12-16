#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_47 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Add)]] X_T38 = add(X_T37, X_I_47)
// Elementwise op: X_T39 = cmp_lt(X_T38, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T40 = cond(X_T39, X_T3, X_T38)
// Elementwise op: [[pid(Sqrt)]] X_T41 = sqrt(X_T40)
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
__kernel void kernel_c42_sdk_1(__global float* restrict  X_T41, __global const float* restrict  X_I_47)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_47 = X_I_47[i1_tid];
  float LX_T38 = (0.0010000000474974513f + LX_I_47);
  int LX_T39 = (LX_T38 < (float)0);
  float LX_T40 = select((float)LX_T38, (float)0, (int)LX_T39);
  float LX_T41 = native_sqrt(LX_T40);
  X_T41[i1_tid] = LX_T41;
}
