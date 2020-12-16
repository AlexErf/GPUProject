#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 32 1 1
// lid: 32 1 1
// Names: { i1 }
// Ranges: { 32 }
// Out stride: { 1 }
// Elementwise input X_I_112 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Add)]] X_T77 = add(X_T76, X_I_112)
// Elementwise op: X_T78 = cmp_lt(X_T77, X_T6)
// Elementwise op: [[pid(Sqrt)]] X_T79 = cond(X_T78, X_T6, X_T77)
// Elementwise op: [[pid(Sqrt)]] X_T80 = sqrt(X_T79)
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
__kernel void kernel_c25_sdk_16(__global float* restrict  X_T80, __global const float* restrict  X_I_112)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 32);
  float LX_I_112 = X_I_112[i1_tid];
  float LX_T77 = (0.0010000000474974513f + LX_I_112);
  int LX_T78 = (LX_T77 < (float)0);
  float LX_T79 = select((float)LX_T77, (float)0, (int)LX_T78);
  float LX_T80 = native_sqrt(LX_T79);
  X_T80[i1_tid] = LX_T80;
}
