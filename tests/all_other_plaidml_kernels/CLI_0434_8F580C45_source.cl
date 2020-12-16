#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1088 }
// Out stride: { 1 }
// Elementwise input X_I_473 shape: fp32(1088):(1):4.25 KiB
// Elementwise op: [[pid(Add)]] X_T1226 = add(X_T63, X_I_473)
// Elementwise op: X_T1227 = cmp_lt(X_T1226, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1228 = cond(X_T1227, X_T36, X_T1226)
// Elementwise op: [[pid(Sqrt)]] X_T1229 = sqrt(X_T1228)
// Tile size: { 256 }
// Contraction output var shape: fp32(1088):(1):4.25 KiB
// Computed true ops: 4352
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
__kernel void kernel_c108_sdk_415(__global float* restrict  X_T1229, __global const float* restrict  X_I_473)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_473 = X_I_473[gout_idx];
    float LX_T1226 = (1.0009999641624745e-5f + LX_I_473);
    int LX_T1227 = (LX_T1226 < (float)0);
    float LX_T1228 = select((float)LX_T1226, (float)0, (int)LX_T1227);
    float LX_T1229 = native_sqrt(LX_T1228);
    X_T1229[gout_idx] = LX_T1229;
  }
}
