#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_157 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T402 = add(X_T33, X_I_157)
// Elementwise op: X_T403 = cmp_lt(X_T402, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T404 = cond(X_T403, X_T4, X_T402)
// Elementwise op: [[pid(Sqrt)]] X_T405 = sqrt(X_T404)
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
__kernel void kernel_c56_sdk_133(__global float* restrict  X_T405, __global const float* restrict  X_I_157)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_157 = X_I_157[i1_tid];
  float LX_T402 = (0.0010000000474974513f + LX_I_157);
  int LX_T403 = (LX_T402 < (float)0);
  float LX_T404 = select((float)LX_T402, (float)0, (int)LX_T403);
  float LX_T405 = native_sqrt(LX_T404);
  X_T405[i1_tid] = LX_T405;
}
