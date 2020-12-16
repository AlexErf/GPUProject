#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1024 }
// Out stride: { 1 }
// Elementwise input X_I_453 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Add)]] X_T1184 = add(X_T71, X_I_453)
// Elementwise op: X_T1185 = cmp_lt(X_T1184, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1186 = cond(X_T1185, X_T36, X_T1184)
// Elementwise op: [[pid(Sqrt)]] X_T1187 = sqrt(X_T1186)
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
__kernel void kernel_c124_sdk_397(__global float* restrict  X_T1187, __global const float* restrict  X_I_453)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_453 = X_I_453[gout_idx];
  float LX_T1184 = (1.0009999641624745e-5f + LX_I_453);
  int LX_T1185 = (LX_T1184 < (float)0);
  float LX_T1186 = select((float)LX_T1184, (float)0, (int)LX_T1185);
  float LX_T1187 = native_sqrt(LX_T1186);
  X_T1187[gout_idx] = LX_T1187;
}
