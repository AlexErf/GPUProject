#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 288 }
// Out stride: { 83232, 4896, 288, 1 }
// Elementwise input X_T1977 shape: fp32(1, 17, 17, 288):(83232, 4896, 288, 1):325.125 KiB
// Elementwise input X_T1981 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_709 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1982 = div(X_T1977, X_T1981)
// Elementwise op: [[pid(Add, Switch)]] X_T1983 = add(X_T1982, X_I_709)
// Elementwise op: X_T1984 = cmp_lt(X_T1983, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1985 = cond(X_T1984, X_T2, X_T1983)
// Tile size: { 1, 2, 17, 32 }
// Contraction output var shape: fp32(1, 17, 17, 288):(83232, 4896, 288, 1):325.125 KiB
// Computed true ops: 332928
// Computed work groups: 81
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 408
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 9, 1
__kernel void kernel_c51_sdk_646(__global float* restrict  X_T1985, __global const float* restrict  X_T1977, __global const float* restrict  X_T1981, __global const float* restrict  X_I_709)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i2_cond = ((i2_gid != 16) || (i2_tid < 1));
  if (i2_cond)
  {
    for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
    {
      int i3_cond = ((i3_lid < 4) || (i3_tid < 1));
      if (i3_cond)
      {
        int i3 = ((4 * i3_lid) + i3_tid);
        int gout_idx = (((4896 * (i2_gid + i2_tid)) + (288 * i3)) + (i4_gid + i4_tid));
        float LX_T1977 = X_T1977[gout_idx];
        float LX_T1981 = X_T1981[(i4_gid + i4_tid)];
        float LX_I_709 = X_I_709[(i4_gid + i4_tid)];
        float LX_T1982 = (LX_T1977 / LX_T1981);
        float LX_T1983 = (LX_T1982 + LX_I_709);
        int LX_T1984 = (LX_T1983 < 0.0f);
        float LX_T1985 = select((float)LX_T1983, (float)0.0f, (int)LX_T1984);
        X_T1985[gout_idx] = LX_T1985;
      }
    }
  }
}
