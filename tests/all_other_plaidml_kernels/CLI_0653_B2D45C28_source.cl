#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1248 }
// Out stride: { 61152, 8736, 1248, 1 }
// Elementwise input X_T2078 shape: fp32(1, 7, 7, 1248):(61152, 8736, 1248, 1):238.875 KiB
// Elementwise input X_T2082 shape: fp32(1248):(1):4.875 KiB
// Elementwise input X_I_801 shape: fp32(1248):(1):4.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2083 = div(X_T2078, X_T2082)
// Elementwise op: [[pid(Add, Switch)]] X_T2084 = add(X_T2083, X_I_801)
// Elementwise op: X_T2085 = cmp_lt(X_T2084, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2086 = cond(X_T2085, X_T2, X_T2084)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1248):(61152, 8736, 1248, 1):238.875 KiB
// Computed true ops: 244608
// Computed work groups: 70
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 10, 1
__kernel void kernel_c124_sdk_719(__global float* restrict  X_T2086, __global const float* restrict  X_T2078, __global const float* restrict  X_T2082, __global const float* restrict  X_I_801)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 1152));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((8736 * i2_gid) + (1248 * i3_tid)) + (i4_gid + i4));
        float LX_T2078 = X_T2078[gout_idx];
        float LX_T2082 = X_T2082[(i4_gid + i4)];
        float LX_I_801 = X_I_801[(i4_gid + i4)];
        float LX_T2083 = (LX_T2078 / LX_T2082);
        float LX_T2084 = (LX_T2083 + LX_I_801);
        int LX_T2085 = (LX_T2084 < 0.0f);
        float LX_T2086 = select((float)LX_T2084, (float)0.0f, (int)LX_T2085);
        X_T2086[gout_idx] = LX_T2086;
      }
    }
  }
}
