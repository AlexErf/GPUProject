#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1504 }
// Out stride: { 73696, 10528, 1504, 1 }
// Elementwise input X_T2070 shape: fp32(1, 7, 7, 1504):(73696, 10528, 1504, 1):287.875 KiB
// Elementwise input X_T2074 shape: fp32(1504):(1):5.875 KiB
// Elementwise input X_I_801 shape: fp32(1504):(1):5.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2075 = div(X_T2070, X_T2074)
// Elementwise op: [[pid(Add, Switch)]] X_T2076 = add(X_T2075, X_I_801)
// Elementwise op: X_T2077 = cmp_lt(X_T2076, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2078 = cond(X_T2077, X_T2, X_T2076)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1504):(73696, 10528, 1504, 1):287.875 KiB
// Computed true ops: 294784
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c108_sdk_719(__global float* restrict  X_T2078, __global const float* restrict  X_T2070, __global const float* restrict  X_T2074, __global const float* restrict  X_I_801)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 1408));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((10528 * i2_gid) + (1504 * i3_tid)) + (i4_gid + i4));
        float LX_T2070 = X_T2070[gout_idx];
        float LX_T2074 = X_T2074[(i4_gid + i4)];
        float LX_I_801 = X_I_801[(i4_gid + i4)];
        float LX_T2075 = (LX_T2070 / LX_T2074);
        float LX_T2076 = (LX_T2075 + LX_I_801);
        int LX_T2077 = (LX_T2076 < 0.0f);
        float LX_T2078 = select((float)LX_T2076, (float)0.0f, (int)LX_T2077);
        X_T2078[gout_idx] = LX_T2078;
      }
    }
  }
}
