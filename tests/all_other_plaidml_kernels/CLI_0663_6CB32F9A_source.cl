#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1312 }
// Out stride: { 64288, 9184, 1312, 1 }
// Elementwise input X_T2128 shape: fp32(1, 7, 7, 1312):(64288, 9184, 1312, 1):251.125 KiB
// Elementwise input X_T2132 shape: fp32(1312):(1):5.125 KiB
// Elementwise input X_I_821 shape: fp32(1312):(1):5.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2133 = div(X_T2128, X_T2132)
// Elementwise op: [[pid(Add, Switch)]] X_T2134 = add(X_T2133, X_I_821)
// Elementwise op: X_T2135 = cmp_lt(X_T2134, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2136 = cond(X_T2135, X_T2, X_T2134)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1312):(64288, 9184, 1312, 1):251.125 KiB
// Computed true ops: 257152
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
__kernel void kernel_c124_sdk_737(__global float* restrict  X_T2136, __global const float* restrict  X_T2128, __global const float* restrict  X_T2132, __global const float* restrict  X_I_821)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 1280));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((9184 * i2_gid) + (1312 * i3_tid)) + (i4_gid + i4));
        float LX_T2128 = X_T2128[gout_idx];
        float LX_T2132 = X_T2132[(i4_gid + i4)];
        float LX_I_821 = X_I_821[(i4_gid + i4)];
        float LX_T2133 = (LX_T2128 / LX_T2132);
        float LX_T2134 = (LX_T2133 + LX_I_821);
        int LX_T2135 = (LX_T2134 < 0.0f);
        float LX_T2136 = select((float)LX_T2134, (float)0.0f, (int)LX_T2135);
        X_T2136[gout_idx] = LX_T2136;
      }
    }
  }
}
