#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1088 }
// Out stride: { 53312, 7616, 1088, 1 }
// Elementwise input X_T1745 shape: fp32(1, 7, 7, 1088):(53312, 7616, 1088, 1):208.25 KiB
// Elementwise input X_T1749 shape: fp32(1088):(1):4.25 KiB
// Elementwise input X_I_671 shape: fp32(1088):(1):4.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1750 = div(X_T1745, X_T1749)
// Elementwise op: [[pid(Add, Switch)]] X_T1751 = add(X_T1750, X_I_671)
// Elementwise op: X_T1752 = cmp_lt(X_T1751, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1753 = cond(X_T1752, X_T2, X_T1751)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1088):(53312, 7616, 1088, 1):208.25 KiB
// Computed true ops: 213248
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c108_sdk_602(__global float* restrict  X_T1753, __global const float* restrict  X_T1745, __global const float* restrict  X_T1749, __global const float* restrict  X_I_671)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1024));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((7616 * i2_gid) + (1088 * i3_tid)) + (i4_gid + i4));
        float LX_T1745 = X_T1745[gout_idx];
        float LX_T1749 = X_T1749[(i4_gid + i4)];
        float LX_I_671 = X_I_671[(i4_gid + i4)];
        float LX_T1750 = (LX_T1745 / LX_T1749);
        float LX_T1751 = (LX_T1750 + LX_I_671);
        int LX_T1752 = (LX_T1751 < 0.0f);
        float LX_T1753 = select((float)LX_T1751, (float)0.0f, (int)LX_T1752);
        X_T1753[gout_idx] = LX_T1753;
      }
    }
  }
}
