#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1024 }
// Out stride: { 1 }
// Elementwise input X_I_136 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Add)]] X_T409 = add(X_T76, X_I_136)
// Elementwise op: X_T410 = cmp_lt(X_T409, X_T6)
// Elementwise op: [[pid(Sqrt)]] X_T411 = cond(X_T410, X_T6, X_T409)
// Elementwise op: [[pid(Sqrt)]] X_T412 = sqrt(X_T411)
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
__kernel void kernel_c25_sdk_104(__global float* restrict  X_T412, __global const float* restrict  X_I_136)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_136 = X_I_136[gout_idx];
  float LX_T409 = (0.0010000000474974513f + LX_I_136);
  int LX_T410 = (LX_T409 < (float)0);
  float LX_T411 = select((float)LX_T409, (float)0, (int)LX_T410);
  float LX_T412 = native_sqrt(LX_T411);
  X_T412[gout_idx] = LX_T412;
}
