#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_181 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Add)]] X_T106 = add(X_T105, X_I_181)
// Elementwise op: X_T107 = cmp_lt(X_T106, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T108 = cond(X_T107, X_T3, X_T106)
// Elementwise op: [[pid(Sqrt)]] X_T109 = sqrt(X_T108)
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
__kernel void kernel_c28_sdk_35(__global float* restrict  X_T109, __global const float* restrict  X_I_181)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_181 = X_I_181[i1_tid];
  float LX_T106 = (0.0010000000474974513f + LX_I_181);
  int LX_T107 = (LX_T106 < (float)0);
  float LX_T108 = select((float)LX_T106, (float)0, (int)LX_T107);
  float LX_T109 = native_sqrt(LX_T108);
  X_T109[i1_tid] = LX_T109;
}
