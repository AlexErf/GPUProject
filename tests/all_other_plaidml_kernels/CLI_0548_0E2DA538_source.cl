#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1568 }
// Out stride: { 1 }
// Elementwise input X_I_623 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(Add)]] X_T1609 = add(X_T71, X_I_623)
// Elementwise op: X_T1610 = cmp_lt(X_T1609, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1611 = cond(X_T1610, X_T36, X_T1609)
// Elementwise op: [[pid(Sqrt)]] X_T1612 = sqrt(X_T1611)
// Tile size: { 256 }
// Contraction output var shape: fp32(1568):(1):6.125 KiB
// Computed true ops: 6272
// Computed work groups: 7
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 1, 1
__kernel void kernel_c124_sdk_550(__global float* restrict  X_T1612, __global const float* restrict  X_I_623)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 32));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_623 = X_I_623[gout_idx];
    float LX_T1609 = (1.0009999641624745e-5f + LX_I_623);
    int LX_T1610 = (LX_T1609 < (float)0);
    float LX_T1611 = select((float)LX_T1609, (float)0, (int)LX_T1610);
    float LX_T1612 = native_sqrt(LX_T1611);
    X_T1612[gout_idx] = LX_T1612;
  }
}
