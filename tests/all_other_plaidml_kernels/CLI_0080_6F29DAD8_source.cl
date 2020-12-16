#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 96 }
// Out stride: { 1 }
// Elementwise input X_I_94 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Add)]] X_T96 = add(X_T59, X_I_94)
// Elementwise op: X_T97 = cmp_lt(X_T96, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T98 = cond(X_T97, X_T4, X_T96)
// Elementwise op: [[pid(Sqrt)]] X_T99 = sqrt(X_T98)
// Tile size: { 96 }
// Contraction output var shape: fp32(96):(1):384 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 12
// Computed mem write: 384
// Computed operations: 96
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c43_sdk_19(__global float* restrict  X_T99, __global const float* restrict  X_I_94)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  int i1_cond = (i1_tid < 96);
  if (i1_cond)
  {
    float LX_I_94 = X_I_94[i1_tid];
    float LX_T96 = (0.0010000000474974513f + LX_I_94);
    int LX_T97 = (LX_T96 < (float)0);
    float LX_T98 = select((float)LX_T96, (float)0, (int)LX_T97);
    float LX_T99 = native_sqrt(LX_T98);
    X_T99[i1_tid] = LX_T99;
  }
}
