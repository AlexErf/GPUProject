#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_120 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T189 = add(X_T76, X_I_120)
// Elementwise op: X_T190 = cmp_lt(X_T189, X_T6)
// Elementwise op: [[pid(Sqrt)]] X_T191 = cond(X_T190, X_T6, X_T189)
// Elementwise op: [[pid(Sqrt)]] X_T192 = sqrt(X_T191)
// Tile size: { 256 }
// Contraction output var shape: fp32(256):(1):1 KiB
// Computed true ops: 1024
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c25_sdk_46(__global float* restrict  X_T192, __global const float* restrict  X_I_120)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_120 = X_I_120[i1_tid];
  float LX_T189 = (0.0010000000474974513f + LX_I_120);
  int LX_T190 = (LX_T189 < (float)0);
  float LX_T191 = select((float)LX_T189, (float)0, (int)LX_T190);
  float LX_T192 = native_sqrt(LX_T191);
  X_T192[i1_tid] = LX_T192;
}
