#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 160 }
// Out stride: { 1 }
// Elementwise input X_I_228 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Add)]] X_T587 = add(X_T59, X_I_228)
// Elementwise op: X_T588 = cmp_lt(X_T587, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T589 = cond(X_T588, X_T4, X_T587)
// Elementwise op: [[pid(Sqrt)]] X_T590 = sqrt(X_T589)
// Tile size: { 160 }
// Contraction output var shape: fp32(160):(1):640 bytes
// Computed true ops: 640
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 20
// Computed mem write: 640
// Computed operations: 160
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c43_sdk_158(__global float* restrict  X_T590, __global const float* restrict  X_I_228)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 160);
  if (i1_cond)
  {
    float LX_I_228 = X_I_228[i1_tid];
    float LX_T587 = (0.0010000000474974513f + LX_I_228);
    int LX_T588 = (LX_T587 < (float)0);
    float LX_T589 = select((float)LX_T587, (float)0, (int)LX_T588);
    float LX_T590 = native_sqrt(LX_T589);
    X_T590[i1_tid] = LX_T590;
  }
}
