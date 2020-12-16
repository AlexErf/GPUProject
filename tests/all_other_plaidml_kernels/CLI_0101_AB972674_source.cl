#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 84 }
// Out stride: { 1 }
// Elementwise input X_I_121 shape: fp32(84):(1):336 bytes
// Elementwise op: [[pid(Add)]] X_T244 = add(X_T40, X_I_121)
// Elementwise op: X_T245 = cmp_lt(X_T244, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T246 = cond(X_T245, X_T3, X_T244)
// Elementwise op: [[pid(Sqrt)]] X_T247 = sqrt(X_T246)
// Tile size: { 84 }
// Contraction output var shape: fp32(84):(1):336 bytes
// Computed true ops: 336
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 84
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c42_sdk_75(__global float* restrict  X_T247, __global const float* restrict  X_I_121)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 84);
  if (i1_cond)
  {
    float LX_I_121 = X_I_121[i1_tid];
    float LX_T244 = (0.0010000000474974513f + LX_I_121);
    int LX_T245 = (LX_T244 < (float)0);
    float LX_T246 = select((float)LX_T244, (float)0, (int)LX_T245);
    float LX_T247 = native_sqrt(LX_T246);
    X_T247[i1_tid] = LX_T247;
  }
}
