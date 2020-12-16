#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16 1 1
// lid: 16 1 1
// Names: { i1 }
// Ranges: { 11 }
// Out stride: { 1 }
// Elementwise input X_I_48 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(Add)]] X_T49 = add(X_T37, X_I_48)
// Elementwise op: X_T50 = cmp_lt(X_T49, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T51 = cond(X_T50, X_T3, X_T49)
// Elementwise op: [[pid(Sqrt)]] X_T52 = sqrt(X_T51)
// Tile size: { 11 }
// Contraction output var shape: fp32(11):(1):44 bytes
// Computed true ops: 44
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 11
// Computed rollups: 0
// Computed threads used: 16
// lwork = 16, 1, 1
// gwork = 16, 1, 1
__kernel void kernel_c42_sdk_4(__global float* restrict  X_T52, __global const float* restrict  X_I_48)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 16);
  int i1_cond = (i1_tid < 11);
  if (i1_cond)
  {
    float LX_I_48 = X_I_48[i1_tid];
    float LX_T49 = (0.0010000000474974513f + LX_I_48);
    int LX_T50 = (LX_T49 < (float)0);
    float LX_T51 = select((float)LX_T49, (float)0, (int)LX_T50);
    float LX_T52 = native_sqrt(LX_T51);
    X_T52[i1_tid] = LX_T52;
  }
}
