#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 768 }
// Out stride: { 1 }
// Elementwise input X_I_373 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(Add)]] X_T956 = add(X_T43, X_I_373)
// Elementwise op: X_T957 = cmp_lt(X_T956, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T958 = cond(X_T957, X_T20, X_T956)
// Elementwise op: [[pid(Sqrt)]] X_T959 = sqrt(X_T958)
// Tile size: { 256 }
// Contraction output var shape: fp32(768):(1):3 KiB
// Computed true ops: 3072
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c68_sdk_325(__global float* restrict  X_T959, __global const float* restrict  X_I_373)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_373 = X_I_373[gout_idx];
  float LX_T956 = (1.0009999641624745e-5f + LX_I_373);
  int LX_T957 = (LX_T956 < (float)0);
  float LX_T958 = select((float)LX_T956, (float)0, (int)LX_T957);
  float LX_T959 = native_sqrt(LX_T958);
  X_T959[gout_idx] = LX_T959;
}
