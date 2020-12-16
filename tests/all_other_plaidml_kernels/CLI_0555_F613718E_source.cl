#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 50 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1600 }
// Out stride: { 313600, 22400, 1600, 1 }
// Elementwise input X_T1633 shape: fp32(1, 14, 14, 1600):(313600, 22400, 1600, 1):1225 KiB
// Elementwise input X_T1637 shape: fp32(1600):(1):6.25 KiB
// Elementwise input X_I_630 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1638 = div(X_T1633, X_T1637)
// Elementwise op: [[pid(Add, Switch)]] X_T1639 = add(X_T1638, X_I_630)
// Elementwise op: X_T1640 = cmp_lt(X_T1639, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1641 = cond(X_T1640, X_T2, X_T1639)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1600):(313600, 22400, 1600, 1):1225 KiB
// Computed true ops: 1254400
// Computed work groups: 350
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 50, 1
__kernel void kernel_c124_sdk_560(__global float* restrict  X_T1641, __global const float* restrict  X_T1633, __global const float* restrict  X_T1637, __global const float* restrict  X_I_630)
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
      int gout_idx = (((22400 * (i2_gid + i2_tid)) + (1600 * i3)) + (i4_gid + i4_tid));
      float LX_T1633 = X_T1633[gout_idx];
      float LX_T1637 = X_T1637[(i4_gid + i4_tid)];
      float LX_I_630 = X_I_630[(i4_gid + i4_tid)];
      float LX_T1638 = (LX_T1633 / LX_T1637);
      float LX_T1639 = (LX_T1638 + LX_I_630);
      int LX_T1640 = (LX_T1639 < 0.0f);
      float LX_T1641 = select((float)LX_T1639, (float)0.0f, (int)LX_T1640);
      X_T1641[gout_idx] = LX_T1641;
    }
  }
}
