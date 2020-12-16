#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_183 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T130 = add(X_T105, X_I_183)
// Elementwise op: X_T131 = cmp_lt(X_T130, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T132 = cond(X_T131, X_T3, X_T130)
// Elementwise op: [[pid(Sqrt)]] X_T133 = sqrt(X_T132)
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
__kernel void kernel_c28_sdk_42(__global float* restrict  X_T133, __global const float* restrict  X_I_183)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_183 = X_I_183[i1_tid];
  float LX_T130 = (0.0010000000474974513f + LX_I_183);
  int LX_T131 = (LX_T130 < (float)0);
  float LX_T132 = select((float)LX_T130, (float)0, (int)LX_T131);
  float LX_T133 = native_sqrt(LX_T132);
  X_T133[i1_tid] = LX_T133;
}
