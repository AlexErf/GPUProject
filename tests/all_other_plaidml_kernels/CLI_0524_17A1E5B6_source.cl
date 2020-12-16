#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1440 }
// Out stride: { 1 }
// Elementwise input X_I_583 shape: fp32(1440):(1):5.625 KiB
// Elementwise op: [[pid(Add)]] X_T1509 = add(X_T71, X_I_583)
// Elementwise op: X_T1510 = cmp_lt(X_T1509, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1511 = cond(X_T1510, X_T36, X_T1509)
// Elementwise op: [[pid(Sqrt)]] X_T1512 = sqrt(X_T1511)
// Tile size: { 256 }
// Contraction output var shape: fp32(1440):(1):5.625 KiB
// Computed true ops: 5760
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
__kernel void kernel_c124_sdk_514(__global float* restrict  X_T1512, __global const float* restrict  X_I_583)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1280) || (i1_tid < 160));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_583 = X_I_583[gout_idx];
    float LX_T1509 = (1.0009999641624745e-5f + LX_I_583);
    int LX_T1510 = (LX_T1509 < (float)0);
    float LX_T1511 = select((float)LX_T1509, (float)0, (int)LX_T1510);
    float LX_T1512 = native_sqrt(LX_T1511);
    X_T1512[gout_idx] = LX_T1512;
  }
}
