#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1184 }
// Out stride: { 1 }
// Elementwise input X_I_503 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(Add)]] X_T1309 = add(X_T71, X_I_503)
// Elementwise op: X_T1310 = cmp_lt(X_T1309, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1311 = cond(X_T1310, X_T36, X_T1309)
// Elementwise op: [[pid(Sqrt)]] X_T1312 = sqrt(X_T1311)
// Tile size: { 256 }
// Contraction output var shape: fp32(1184):(1):4.625 KiB
// Computed true ops: 4736
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
__kernel void kernel_c124_sdk_442(__global float* restrict  X_T1312, __global const float* restrict  X_I_503)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1024) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_503 = X_I_503[gout_idx];
    float LX_T1309 = (1.0009999641624745e-5f + LX_I_503);
    int LX_T1310 = (LX_T1309 < (float)0);
    float LX_T1311 = select((float)LX_T1309, (float)0, (int)LX_T1310);
    float LX_T1312 = native_sqrt(LX_T1311);
    X_T1312[gout_idx] = LX_T1312;
  }
}
