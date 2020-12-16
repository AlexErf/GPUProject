#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 51 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1632 }
// Out stride: { 319872, 22848, 1632, 1 }
// Elementwise input X_T1658 shape: fp32(1, 14, 14, 1632):(319872, 22848, 1632, 1):1249.5 KiB
// Elementwise input X_T1662 shape: fp32(1632):(1):6.375 KiB
// Elementwise input X_I_640 shape: fp32(1632):(1):6.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1663 = div(X_T1658, X_T1662)
// Elementwise op: [[pid(Add, Switch)]] X_T1664 = add(X_T1663, X_I_640)
// Elementwise op: X_T1665 = cmp_lt(X_T1664, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1666 = cond(X_T1665, X_T2, X_T1664)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1632):(319872, 22848, 1632, 1):1249.5 KiB
// Computed true ops: 1279488
// Computed work groups: 357
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 51, 1
__kernel void kernel_c124_sdk_569(__global float* restrict  X_T1666, __global const float* restrict  X_T1658, __global const float* restrict  X_T1662, __global const float* restrict  X_I_640)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((22848 * (i2_gid + i2_tid)) + (1632 * i3)) + (i4_gid + i4_tid));
      float LX_T1658 = X_T1658[gout_idx];
      float LX_T1662 = X_T1662[(i4_gid + i4_tid)];
      float LX_I_640 = X_I_640[(i4_gid + i4_tid)];
      float LX_T1663 = (LX_T1658 / LX_T1662);
      float LX_T1664 = (LX_T1663 + LX_I_640);
      int LX_T1665 = (LX_T1664 < 0.0f);
      float LX_T1666 = select((float)LX_T1664, (float)0.0f, (int)LX_T1665);
      X_T1666[gout_idx] = LX_T1666;
    }
  }
}
