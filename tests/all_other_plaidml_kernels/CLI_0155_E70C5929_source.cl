#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 2 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 256 }
// Out stride: { 73984, 4352, 256, 1 }
// Elementwise input X_T1920 shape: fp32(1, 17, 17, 256):(73984, 4352, 256, 1):289 KiB
// Elementwise input X_T1924 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_8 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1925 = div(X_T1920, X_T1924)
// Elementwise op: [[pid(Add, Switch)]] X_T1926 = add(X_T1925, X_I_8)
// Elementwise op: X_T1927 = cmp_lt(X_T1926, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1928 = cond(X_T1927, X_T2, X_T1926)
// Tile size: { 1, 1, 17, 128 }
// Contraction output var shape: fp32(1, 17, 17, 256):(73984, 4352, 256, 1):289 KiB
// Computed true ops: 295936
// Computed work groups: 34
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 816
// Computed mem write: 8704
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 2, 1
__kernel void kernel_c51_sdk_628(__global float* restrict  X_T1928, __global const float* restrict  X_T1920, __global const float* restrict  X_T1924, __global const float* restrict  X_I_8)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i3_lid = 0; i3_lid < 3; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 2) || (i3_tid < 1));
    if (i3_cond)
    {
      int i3 = ((8 * i3_lid) + i3_tid);
      for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
      {
        int i4 = ((32 * i4_lid) + i4_tid);
        int gout_idx = (((4352 * i2_gid) + (256 * i3)) + (i4_gid + i4));
        float LX_T1920 = X_T1920[gout_idx];
        float LX_T1924 = X_T1924[(i4_gid + i4)];
        float LX_I_8 = X_I_8[(i4_gid + i4)];
        float LX_T1925 = (LX_T1920 / LX_T1924);
        float LX_T1926 = (LX_T1925 + LX_I_8);
        int LX_T1927 = (LX_T1926 < 0.0f);
        float LX_T1928 = select((float)LX_T1926, (float)0.0f, (int)LX_T1927);
        X_T1928[gout_idx] = LX_T1928;
      }
    }
  }
}
