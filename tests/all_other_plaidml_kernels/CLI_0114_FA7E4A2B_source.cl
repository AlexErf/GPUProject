#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 64 }
// Out stride: { 1 }
// Elementwise input X_I_147 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(Add)]] X_T323 = add(X_T59, X_I_147)
// Elementwise op: X_T324 = cmp_lt(X_T323, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T325 = cond(X_T324, X_T4, X_T323)
// Elementwise op: [[pid(Sqrt)]] X_T326 = sqrt(X_T325)
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
__kernel void kernel_c43_sdk_82(__global float* restrict  X_T326, __global const float* restrict  X_I_147)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  float LX_I_147 = X_I_147[i1_tid];
  float LX_T323 = (0.0010000000474974513f + LX_I_147);
  int LX_T324 = (LX_T323 < (float)0);
  float LX_T325 = select((float)LX_T323, (float)0, (int)LX_T324);
  float LX_T326 = native_sqrt(LX_T325);
  X_T326[i1_tid] = LX_T326;
}
