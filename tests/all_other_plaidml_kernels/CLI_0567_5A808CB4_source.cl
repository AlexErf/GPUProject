#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 52 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1664 }
// Out stride: { 326144, 23296, 1664, 1 }
// Elementwise input X_T1683 shape: fp32(1, 14, 14, 1664):(326144, 23296, 1664, 1):1274 KiB
// Elementwise input X_T1687 shape: fp32(1664):(1):6.5 KiB
// Elementwise input X_I_650 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1688 = div(X_T1683, X_T1687)
// Elementwise op: [[pid(Add, Switch)]] X_T1689 = add(X_T1688, X_I_650)
// Elementwise op: X_T1690 = cmp_lt(X_T1689, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1691 = cond(X_T1690, X_T2, X_T1689)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1664):(326144, 23296, 1664, 1):1274 KiB
// Computed true ops: 1304576
// Computed work groups: 364
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 52, 1
__kernel void kernel_c124_sdk_578(__global float* restrict  X_T1691, __global const float* restrict  X_T1683, __global const float* restrict  X_T1687, __global const float* restrict  X_I_650)
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
      int gout_idx = (((23296 * (i2_gid + i2_tid)) + (1664 * i3)) + (i4_gid + i4_tid));
      float LX_T1683 = X_T1683[gout_idx];
      float LX_T1687 = X_T1687[(i4_gid + i4_tid)];
      float LX_I_650 = X_I_650[(i4_gid + i4_tid)];
      float LX_T1688 = (LX_T1683 / LX_T1687);
      float LX_T1689 = (LX_T1688 + LX_I_650);
      int LX_T1690 = (LX_T1689 < 0.0f);
      float LX_T1691 = select((float)LX_T1689, (float)0.0f, (int)LX_T1690);
      X_T1691[gout_idx] = LX_T1691;
    }
  }
}
