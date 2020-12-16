#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1376 }
// Out stride: { 67424, 9632, 1376, 1 }
// Elementwise input X_T1970 shape: fp32(1, 7, 7, 1376):(67424, 9632, 1376, 1):263.375 KiB
// Elementwise input X_T1974 shape: fp32(1376):(1):5.375 KiB
// Elementwise input X_I_761 shape: fp32(1376):(1):5.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1975 = div(X_T1970, X_T1974)
// Elementwise op: [[pid(Add, Switch)]] X_T1976 = add(X_T1975, X_I_761)
// Elementwise op: X_T1977 = cmp_lt(X_T1976, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1978 = cond(X_T1977, X_T2, X_T1976)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1376):(67424, 9632, 1376, 1):263.375 KiB
// Computed true ops: 269696
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c108_sdk_683(__global float* restrict  X_T1978, __global const float* restrict  X_T1970, __global const float* restrict  X_T1974, __global const float* restrict  X_I_761)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 1280));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((9632 * i2_gid) + (1376 * i3_tid)) + (i4_gid + i4));
        float LX_T1970 = X_T1970[gout_idx];
        float LX_T1974 = X_T1974[(i4_gid + i4)];
        float LX_I_761 = X_I_761[(i4_gid + i4)];
        float LX_T1975 = (LX_T1970 / LX_T1974);
        float LX_T1976 = (LX_T1975 + LX_I_761);
        int LX_T1977 = (LX_T1976 < 0.0f);
        float LX_T1978 = select((float)LX_T1976, (float)0.0f, (int)LX_T1977);
        X_T1978[gout_idx] = LX_T1978;
      }
    }
  }
}
