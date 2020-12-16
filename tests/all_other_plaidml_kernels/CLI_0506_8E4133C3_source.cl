#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1344 }
// Out stride: { 1 }
// Elementwise input X_I_553 shape: fp32(1344):(1):5.25 KiB
// Elementwise op: [[pid(Add)]] X_T1434 = add(X_T71, X_I_553)
// Elementwise op: X_T1435 = cmp_lt(X_T1434, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1436 = cond(X_T1435, X_T36, X_T1434)
// Elementwise op: [[pid(Sqrt)]] X_T1437 = sqrt(X_T1436)
// Tile size: { 256 }
// Contraction output var shape: fp32(1344):(1):5.25 KiB
// Computed true ops: 5376
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
__kernel void kernel_c124_sdk_487(__global float* restrict  X_T1437, __global const float* restrict  X_I_553)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1280) || (i1_tid < 64));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_553 = X_I_553[gout_idx];
    float LX_T1434 = (1.0009999641624745e-5f + LX_I_553);
    int LX_T1435 = (LX_T1434 < (float)0);
    float LX_T1436 = select((float)LX_T1434, (float)0, (int)LX_T1435);
    float LX_T1437 = native_sqrt(LX_T1436);
    X_T1437[gout_idx] = LX_T1437;
  }
}
