#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1088 }
// Out stride: { 1 }
// Elementwise input X_I_473 shape: fp32(1088):(1):4.25 KiB
// Elementwise op: [[pid(Add)]] X_T1234 = add(X_T71, X_I_473)
// Elementwise op: X_T1235 = cmp_lt(X_T1234, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1236 = cond(X_T1235, X_T36, X_T1234)
// Elementwise op: [[pid(Sqrt)]] X_T1237 = sqrt(X_T1236)
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
__kernel void kernel_c124_sdk_415(__global float* restrict  X_T1237, __global const float* restrict  X_I_473)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_473 = X_I_473[gout_idx];
    float LX_T1234 = (1.0009999641624745e-5f + LX_I_473);
    int LX_T1235 = (LX_T1234 < (float)0);
    float LX_T1236 = select((float)LX_T1234, (float)0, (int)LX_T1235);
    float LX_T1237 = native_sqrt(LX_T1236);
    X_T1237[gout_idx] = LX_T1237;
  }
}
