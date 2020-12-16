#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 768 }
// Out stride: { 1 }
// Elementwise input X_I_373 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(Add)]] X_T976 = add(X_T63, X_I_373)
// Elementwise op: X_T977 = cmp_lt(X_T976, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T978 = cond(X_T977, X_T36, X_T976)
// Elementwise op: [[pid(Sqrt)]] X_T979 = sqrt(X_T978)
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
__kernel void kernel_c108_sdk_325(__global float* restrict  X_T979, __global const float* restrict  X_I_373)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_373 = X_I_373[gout_idx];
  float LX_T976 = (1.0009999641624745e-5f + LX_I_373);
  int LX_T977 = (LX_T976 < (float)0);
  float LX_T978 = select((float)LX_T976, (float)0, (int)LX_T977);
  float LX_T979 = native_sqrt(LX_T978);
  X_T979[gout_idx] = LX_T979;
}
