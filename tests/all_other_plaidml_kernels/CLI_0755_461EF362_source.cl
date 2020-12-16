#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1888 }
// Out stride: { 1 }
// Elementwise input X_I_1004 shape: fp32(1888):(1):7.375 KiB
// Elementwise op: [[pid(Add)]] X_T2579 = add(X_T71, X_I_1004)
// Elementwise op: X_T2580 = cmp_lt(X_T2579, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T2581 = cond(X_T2580, X_T36, X_T2579)
// Elementwise op: [[pid(Sqrt)]] X_T2582 = sqrt(X_T2581)
// Tile size: { 256 }
// Contraction output var shape: fp32(1888):(1):7.375 KiB
// Computed true ops: 7552
// Computed work groups: 8
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2048, 1, 1
__kernel void kernel_c124_sdk_898(__global float* restrict  X_T2582, __global const float* restrict  X_I_1004)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1792) || (i1_tid < 96));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_1004 = X_I_1004[gout_idx];
    float LX_T2579 = (1.0009999641624745e-5f + LX_I_1004);
    int LX_T2580 = (LX_T2579 < (float)0);
    float LX_T2581 = select((float)LX_T2579, (float)0, (int)LX_T2580);
    float LX_T2582 = native_sqrt(LX_T2581);
    X_T2582[gout_idx] = LX_T2582;
  }
}
