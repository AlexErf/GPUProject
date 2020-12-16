#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 2048 }
// Out stride: { 1 }
// Elementwise input X_I_236 shape: fp32(2048):(1):8 KiB
// Elementwise op: [[pid(Add)]] X_T625 = add(X_T105, X_I_236)
// Elementwise op: X_T626 = cmp_lt(X_T625, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T627 = cond(X_T626, X_T3, X_T625)
// Elementwise op: [[pid(Sqrt)]] X_T628 = sqrt(X_T627)
// Tile size: { 256 }
// Contraction output var shape: fp32(2048):(1):8 KiB
// Computed true ops: 8192
// Computed work groups: 8
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2048, 1, 1
__kernel void kernel_c28_sdk_190(__global float* restrict  X_T628, __global const float* restrict  X_I_236)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_236 = X_I_236[gout_idx];
  float LX_T625 = (0.0010000000474974513f + LX_I_236);
  int LX_T626 = (LX_T625 < (float)0);
  float LX_T627 = select((float)LX_T625, (float)0, (int)LX_T626);
  float LX_T628 = native_sqrt(LX_T627);
  X_T628[gout_idx] = LX_T628;
}
