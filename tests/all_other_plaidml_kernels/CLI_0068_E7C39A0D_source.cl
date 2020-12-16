#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 64 1 1
// lid: 64 1 1
// Names: { i1 }
// Ranges: { 42 }
// Out stride: { 1 }
// Elementwise input X_I_60 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(Add)]] X_T52 = add(X_T40, X_I_60)
// Elementwise op: X_T53 = cmp_lt(X_T52, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T54 = cond(X_T53, X_T3, X_T52)
// Elementwise op: [[pid(Sqrt)]] X_T55 = sqrt(X_T54)
// Tile size: { 42 }
// Contraction output var shape: fp32(42):(1):168 bytes
// Computed true ops: 168
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 8
// Computed mem write: 256
// Computed operations: 42
// Computed rollups: 0
// Computed threads used: 64
// lwork = 64, 1, 1
// gwork = 64, 1, 1
__kernel void kernel_c42_sdk_4(__global float* restrict  X_T55, __global const float* restrict  X_I_60)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 64);
  int i1_cond = (i1_tid < 42);
  if (i1_cond)
  {
    float LX_I_60 = X_I_60[i1_tid];
    float LX_T52 = (0.0010000000474974513f + LX_I_60);
    int LX_T53 = (LX_T52 < (float)0);
    float LX_T54 = select((float)LX_T52, (float)0, (int)LX_T53);
    float LX_T55 = native_sqrt(LX_T54);
    X_T55[i1_tid] = LX_T55;
  }
}
