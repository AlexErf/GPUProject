#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1376 }
// Out stride: { 1 }
// Elementwise input X_I_563 shape: fp32(1376):(1):5.375 KiB
// Elementwise op: [[pid(Add)]] X_T1459 = add(X_T71, X_I_563)
// Elementwise op: X_T1460 = cmp_lt(X_T1459, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1461 = cond(X_T1460, X_T36, X_T1459)
// Elementwise op: [[pid(Sqrt)]] X_T1462 = sqrt(X_T1461)
// Tile size: { 256 }
// Contraction output var shape: fp32(1376):(1):5.375 KiB
// Computed true ops: 5504
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
__kernel void kernel_c124_sdk_496(__global float* restrict  X_T1462, __global const float* restrict  X_I_563)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1280) || (i1_tid < 96));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_563 = X_I_563[gout_idx];
    float LX_T1459 = (1.0009999641624745e-5f + LX_I_563);
    int LX_T1460 = (LX_T1459 < (float)0);
    float LX_T1461 = select((float)LX_T1459, (float)0, (int)LX_T1460);
    float LX_T1462 = native_sqrt(LX_T1461);
    X_T1462[gout_idx] = LX_T1462;
  }
}
