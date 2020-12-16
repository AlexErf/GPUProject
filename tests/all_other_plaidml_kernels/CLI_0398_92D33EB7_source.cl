#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 768 }
// Out stride: { 1 }
// Elementwise input X_I_373 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(Add)]] X_T984 = add(X_T71, X_I_373)
// Elementwise op: X_T985 = cmp_lt(X_T984, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T986 = cond(X_T985, X_T36, X_T984)
// Elementwise op: [[pid(Sqrt)]] X_T987 = sqrt(X_T986)
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
__kernel void kernel_c124_sdk_325(__global float* restrict  X_T987, __global const float* restrict  X_I_373)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_373 = X_I_373[gout_idx];
  float LX_T984 = (1.0009999641624745e-5f + LX_I_373);
  int LX_T985 = (LX_T984 < (float)0);
  float LX_T986 = select((float)LX_T984, (float)0, (int)LX_T985);
  float LX_T987 = native_sqrt(LX_T986);
  X_T987[gout_idx] = LX_T987;
}
