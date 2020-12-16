#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1280 }
// Out stride: { 1 }
// Elementwise input X_I_525 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(Add)]] X_T1375 = add(X_T63, X_I_525)
// Elementwise op: X_T1376 = cmp_lt(X_T1375, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1377 = cond(X_T1376, X_T36, X_T1375)
// Elementwise op: [[pid(Sqrt)]] X_T1378 = sqrt(X_T1377)
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
__kernel void kernel_c108_sdk_468(__global float* restrict  X_T1378, __global const float* restrict  X_I_525)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_525 = X_I_525[gout_idx];
  float LX_T1375 = (1.0009999641624745e-5f + LX_I_525);
  int LX_T1376 = (LX_T1375 < (float)0);
  float LX_T1377 = select((float)LX_T1375, (float)0, (int)LX_T1376);
  float LX_T1378 = native_sqrt(LX_T1377);
  X_T1378[gout_idx] = LX_T1378;
}
