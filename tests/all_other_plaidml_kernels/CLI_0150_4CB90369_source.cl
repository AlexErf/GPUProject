#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1280 }
// Out stride: { 1 }
// Elementwise input X_I_262 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(Add)]] X_T708 = add(X_T59, X_I_262)
// Elementwise op: X_T709 = cmp_lt(X_T708, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T710 = cond(X_T709, X_T4, X_T708)
// Elementwise op: [[pid(Sqrt)]] X_T711 = sqrt(X_T710)
// Tile size: { 256 }
// Contraction output var shape: fp32(1280):(1):5 KiB
// Computed true ops: 5120
// Computed work groups: 5
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 1, 1
__kernel void kernel_c43_sdk_193(__global float* restrict  X_T711, __global const float* restrict  X_I_262)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_262 = X_I_262[gout_idx];
  float LX_T708 = (0.0010000000474974513f + LX_I_262);
  int LX_T709 = (LX_T708 < (float)0);
  float LX_T710 = select((float)LX_T708, (float)0, (int)LX_T709);
  float LX_T711 = native_sqrt(LX_T710);
  X_T711[gout_idx] = LX_T711;
}
