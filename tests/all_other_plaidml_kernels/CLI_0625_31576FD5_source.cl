#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1536 }
// Out stride: { 1 }
// Elementwise input X_I_814 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(Add)]] X_T2096 = add(X_T63, X_I_814)
// Elementwise op: X_T2097 = cmp_lt(X_T2096, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2098 = cond(X_T2097, X_T36, X_T2096)
// Elementwise op: [[pid(Sqrt)]] X_T2099 = sqrt(X_T2098)
// Tile size: { 256 }
// Contraction output var shape: fp32(1536):(1):6 KiB
// Computed true ops: 6144
// Computed work groups: 6
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 1, 1
__kernel void kernel_c108_sdk_727(__global float* restrict  X_T2099, __global const float* restrict  X_I_814)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_814 = X_I_814[gout_idx];
  float LX_T2096 = (1.0009999641624745e-5f + LX_I_814);
  int LX_T2097 = (LX_T2096 < (float)0);
  float LX_T2098 = select((float)LX_T2096, (float)0, (int)LX_T2097);
  float LX_T2099 = native_sqrt(LX_T2098);
  X_T2099[gout_idx] = LX_T2099;
}
