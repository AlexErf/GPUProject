#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3328 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 1632 }
// Out stride: { 1 }
// Elementwise input X_I_643 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(Add)]] X_T1659 = add(X_T71, X_I_643)
// Elementwise op: X_T1660 = cmp_lt(X_T1659, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T1661 = cond(X_T1660, X_T36, X_T1659)
// Elementwise op: [[pid(Sqrt)]] X_T1662 = sqrt(X_T1661)
// Tile size: { 128 }
// Contraction output var shape: fp32(1632):(1):6.375 KiB
// Computed true ops: 6528
// Computed work groups: 13
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 3328, 1, 1
__kernel void kernel_c124_sdk_568(__global float* restrict  X_T1662, __global const float* restrict  X_I_643)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 1536) || (i1_tid < 96));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_643 = X_I_643[gout_idx];
      float LX_T1659 = (1.0009999641624745e-5f + LX_I_643);
      int LX_T1660 = (LX_T1659 < (float)0);
      float LX_T1661 = select((float)LX_T1659, (float)0, (int)LX_T1660);
      float LX_T1662 = native_sqrt(LX_T1661);
      X_T1662[gout_idx] = LX_T1662;
    }
  }
}
