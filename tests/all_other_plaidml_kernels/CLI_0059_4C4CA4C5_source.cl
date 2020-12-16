#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1024 }
// Out stride: { 1 }
// Elementwise input X_I_284 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Add)]] X_T356 = add(X_T37, X_I_284)
// Elementwise op: X_T357 = cmp_lt(X_T356, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T358 = cond(X_T357, X_T3, X_T356)
// Elementwise op: [[pid(Sqrt)]] X_T359 = sqrt(X_T358)
// Tile size: { 256 }
// Contraction output var shape: fp32(1024):(1):4 KiB
// Computed true ops: 4096
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c29_sdk_82(__global float* restrict  X_T359, __global const float* restrict  X_I_284)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_284 = X_I_284[gout_idx];
  float LX_T356 = (0.0010000000474974513f + LX_I_284);
  int LX_T357 = (LX_T356 < (float)0);
  float LX_T358 = select((float)LX_T356, (float)0, (int)LX_T357);
  float LX_T359 = native_sqrt(LX_T358);
  X_T359[gout_idx] = LX_T359;
}
