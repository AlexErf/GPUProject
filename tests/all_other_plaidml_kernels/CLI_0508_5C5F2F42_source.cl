#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 832 }
// Out stride: { 40768, 5824, 832, 1 }
// Elementwise input X_T1545 shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Elementwise input X_T1549 shape: fp32(832):(1):3.25 KiB
// Elementwise input X_I_591 shape: fp32(832):(1):3.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1550 = div(X_T1545, X_T1549)
// Elementwise op: [[pid(Add, Switch)]] X_T1551 = add(X_T1550, X_I_591)
// Elementwise op: X_T1552 = cmp_lt(X_T1551, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1553 = cond(X_T1552, X_T2, X_T1551)
// Tile size: { 1, 1, 1, 832 }
// Contraction output var shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Computed true ops: 163072
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 312
// Computed mem write: 3328
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_530(__global float* restrict  X_T1553, __global const float* restrict  X_T1545, __global const float* restrict  X_T1549, __global const float* restrict  X_I_591)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((5824 * i2_gid) + (832 * i3_gid)) + i4);
      float LX_T1545 = X_T1545[gout_idx];
      float LX_T1549 = X_T1549[i4];
      float LX_I_591 = X_I_591[i4];
      float LX_T1550 = (LX_T1545 / LX_T1549);
      float LX_T1551 = (LX_T1550 + LX_I_591);
      int LX_T1552 = (LX_T1551 < 0.0f);
      float LX_T1553 = select((float)LX_T1551, (float)0.0f, (int)LX_T1552);
      X_T1553[gout_idx] = LX_T1553;
    }
  }
}
