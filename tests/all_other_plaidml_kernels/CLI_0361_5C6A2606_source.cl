#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1024 }
// Out stride: { 1 }
// Elementwise input X_I_445 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Add)]] X_T1155 = add(X_T43, X_I_445)
// Elementwise op: X_T1156 = cmp_lt(X_T1155, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T1157 = cond(X_T1156, X_T20, X_T1155)
// Elementwise op: [[pid(Sqrt)]] X_T1158 = sqrt(X_T1157)
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
__kernel void kernel_c68_sdk_396(__global float* restrict  X_T1158, __global const float* restrict  X_I_445)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_445 = X_I_445[gout_idx];
  float LX_T1155 = (1.0009999641624745e-5f + LX_I_445);
  int LX_T1156 = (LX_T1155 < (float)0);
  float LX_T1157 = select((float)LX_T1155, (float)0, (int)LX_T1156);
  float LX_T1158 = native_sqrt(LX_T1157);
  X_T1158[gout_idx] = LX_T1158;
}
