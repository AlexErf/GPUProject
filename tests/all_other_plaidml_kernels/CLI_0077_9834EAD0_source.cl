#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16 1 1
// lid: 16 1 1
// Names: { i1 }
// Ranges: { 16 }
// Out stride: { 1 }
// Elementwise input X_I_93 shape: fp32(16):(1):64 bytes
// Elementwise op: [[pid(Add)]] X_T87 = add(X_T59, X_I_93)
// Elementwise op: X_T88 = cmp_lt(X_T87, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T89 = cond(X_T88, X_T4, X_T87)
// Elementwise op: [[pid(Sqrt)]] X_T90 = sqrt(X_T89)
// Tile size: { 16 }
// Contraction output var shape: fp32(16):(1):64 bytes
// Computed true ops: 64
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 4
// Computed mem write: 128
// Computed operations: 16
// Computed rollups: 0
// Computed threads used: 16
// lwork = 16, 1, 1
// gwork = 16, 1, 1
__kernel void kernel_c43_sdk_16(__global float* restrict  X_T90, __global const float* restrict  X_I_93)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 16);
  float LX_I_93 = X_I_93[i1_tid];
  float LX_T87 = (0.0010000000474974513f + LX_I_93);
  int LX_T88 = (LX_T87 < (float)0);
  float LX_T89 = select((float)LX_T87, (float)0, (int)LX_T88);
  float LX_T90 = native_sqrt(LX_T89);
  X_T90[i1_tid] = LX_T90;
}
