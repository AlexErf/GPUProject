#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1344 }
// Out stride: { 65856, 9408, 1344, 1 }
// Elementwise input X_T2153 shape: fp32(1, 7, 7, 1344):(65856, 9408, 1344, 1):257.25 KiB
// Elementwise input X_T2157 shape: fp32(1344):(1):5.25 KiB
// Elementwise input X_I_831 shape: fp32(1344):(1):5.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2158 = div(X_T2153, X_T2157)
// Elementwise op: [[pid(Add, Switch)]] X_T2159 = add(X_T2158, X_I_831)
// Elementwise op: X_T2160 = cmp_lt(X_T2159, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2161 = cond(X_T2160, X_T2, X_T2159)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1344):(65856, 9408, 1344, 1):257.25 KiB
// Computed true ops: 263424
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
__kernel void kernel_c124_sdk_746(__global float* restrict  X_T2161, __global const float* restrict  X_T2153, __global const float* restrict  X_T2157, __global const float* restrict  X_I_831)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1280));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((9408 * i2_gid) + (1344 * i3_tid)) + (i4_gid + i4));
        float LX_T2153 = X_T2153[gout_idx];
        float LX_T2157 = X_T2157[(i4_gid + i4)];
        float LX_I_831 = X_I_831[(i4_gid + i4)];
        float LX_T2158 = (LX_T2153 / LX_T2157);
        float LX_T2159 = (LX_T2158 + LX_I_831);
        int LX_T2160 = (LX_T2159 < 0.0f);
        float LX_T2161 = select((float)LX_T2159, (float)0.0f, (int)LX_T2160);
        X_T2161[gout_idx] = LX_T2161;
      }
    }
  }
}
