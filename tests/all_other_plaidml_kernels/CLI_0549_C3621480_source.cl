#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 49 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1568 }
// Out stride: { 307328, 21952, 1568, 1 }
// Elementwise input X_T1608 shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Elementwise input X_T1612 shape: fp32(1568):(1):6.125 KiB
// Elementwise input X_I_620 shape: fp32(1568):(1):6.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1613 = div(X_T1608, X_T1612)
// Elementwise op: [[pid(Add, Switch)]] X_T1614 = add(X_T1613, X_I_620)
// Elementwise op: X_T1615 = cmp_lt(X_T1614, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1616 = cond(X_T1615, X_T2, X_T1614)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Computed true ops: 1229312
// Computed work groups: 343
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 49, 1
__kernel void kernel_c124_sdk_551(__global float* restrict  X_T1616, __global const float* restrict  X_T1608, __global const float* restrict  X_T1612, __global const float* restrict  X_I_620)
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
      int gout_idx = (((21952 * (i2_gid + i2_tid)) + (1568 * i3)) + (i4_gid + i4_tid));
      float LX_T1608 = X_T1608[gout_idx];
      float LX_T1612 = X_T1612[(i4_gid + i4_tid)];
      float LX_I_620 = X_I_620[(i4_gid + i4_tid)];
      float LX_T1613 = (LX_T1608 / LX_T1612);
      float LX_T1614 = (LX_T1613 + LX_I_620);
      int LX_T1615 = (LX_T1614 < 0.0f);
      float LX_T1616 = select((float)LX_T1614, (float)0.0f, (int)LX_T1615);
      X_T1616[gout_idx] = LX_T1616;
    }
  }
}
