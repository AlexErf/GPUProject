#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1376 }
// Out stride: { 67424, 9632, 1376, 1 }
// Elementwise input X_T2178 shape: fp32(1, 7, 7, 1376):(67424, 9632, 1376, 1):263.375 KiB
// Elementwise input X_T2182 shape: fp32(1376):(1):5.375 KiB
// Elementwise input X_I_841 shape: fp32(1376):(1):5.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2183 = div(X_T2178, X_T2182)
// Elementwise op: [[pid(Add, Switch)]] X_T2184 = add(X_T2183, X_I_841)
// Elementwise op: X_T2185 = cmp_lt(X_T2184, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2186 = cond(X_T2185, X_T2, X_T2184)
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
__kernel void kernel_c124_sdk_755(__global float* restrict  X_T2186, __global const float* restrict  X_T2178, __global const float* restrict  X_T2182, __global const float* restrict  X_I_841)
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
        float LX_T2178 = X_T2178[gout_idx];
        float LX_T2182 = X_T2182[(i4_gid + i4)];
        float LX_I_841 = X_I_841[(i4_gid + i4)];
        float LX_T2183 = (LX_T2178 / LX_T2182);
        float LX_T2184 = (LX_T2183 + LX_I_841);
        int LX_T2185 = (LX_T2184 < 0.0f);
        float LX_T2186 = select((float)LX_T2184, (float)0.0f, (int)LX_T2185);
        X_T2186[gout_idx] = LX_T2186;
      }
    }
  }
}
