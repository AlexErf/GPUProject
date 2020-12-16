#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 336 }
// Out stride: { 1 }
// Elementwise input X_I_597 shape: fp32(336):(1):1.3125 KiB
// Elementwise op: [[pid(Add)]] X_T1551 = add(X_T40, X_I_597)
// Elementwise op: X_T1552 = cmp_lt(X_T1551, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T1553 = cond(X_T1552, X_T3, X_T1551)
// Elementwise op: [[pid(Sqrt)]] X_T1554 = sqrt(X_T1553)
// Tile size: { 128 }
// Contraction output var shape: fp32(336):(1):1.3125 KiB
// Computed true ops: 1344
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c42_sdk_591(__global float* restrict  X_T1554, __global const float* restrict  X_I_597)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 128);
  int i1_tid = (tid % 128);
  int i1_cond = ((i1_gid != 256) || (i1_tid < 80));
  if (i1_cond)
  {
    if ((tid < 128))
    {
      int gout_idx = (i1_gid + i1_tid);
      float LX_I_597 = X_I_597[gout_idx];
      float LX_T1551 = (0.0010000000474974513f + LX_I_597);
      int LX_T1552 = (LX_T1551 < (float)0);
      float LX_T1553 = select((float)LX_T1551, (float)0, (int)LX_T1552);
      float LX_T1554 = native_sqrt(LX_T1553);
      X_T1554[gout_idx] = LX_T1554;
    }
  }
}
