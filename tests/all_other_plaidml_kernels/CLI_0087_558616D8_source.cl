#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 24 }
// Out stride: { 1 }
// Elementwise input X_I_96 shape: fp32(24):(1):96 bytes
// Elementwise op: [[pid(Add)]] X_T125 = add(X_T59, X_I_96)
// Elementwise op: X_T126 = cmp_lt(X_T125, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T127 = cond(X_T126, X_T4, X_T125)
// Elementwise op: [[pid(Sqrt)]] X_T128 = sqrt(X_T127)
// Tile size: { 24 }
// Contraction output var shape: fp32(24):(1):96 bytes
// Computed true ops: 96
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 24
// Computed rollups: 0
// Computed threads used: 32
// lwork = 32, 1, 1
// gwork = 32, 1, 1
__kernel void kernel_c43_sdk_27(__global float* restrict  X_T128, __global const float* restrict  X_I_96)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  int i1_cond = (i1_tid < 24);
  if (i1_cond)
  {
    float LX_I_96 = X_I_96[i1_tid];
    float LX_T125 = (0.0010000000474974513f + LX_I_96);
    int LX_T126 = (LX_T125 < (float)0);
    float LX_T127 = select((float)LX_T125, (float)0, (int)LX_T126);
    float LX_T128 = native_sqrt(LX_T127);
    X_T128[i1_tid] = LX_T128;
  }
}
