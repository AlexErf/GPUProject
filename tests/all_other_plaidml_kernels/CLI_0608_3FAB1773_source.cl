#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1440 }
// Out stride: { 70560, 10080, 1440, 1 }
// Elementwise input X_T2020 shape: fp32(1, 7, 7, 1440):(70560, 10080, 1440, 1):275.625 KiB
// Elementwise input X_T2024 shape: fp32(1440):(1):5.625 KiB
// Elementwise input X_I_781 shape: fp32(1440):(1):5.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2025 = div(X_T2020, X_T2024)
// Elementwise op: [[pid(Add, Switch)]] X_T2026 = add(X_T2025, X_I_781)
// Elementwise op: X_T2027 = cmp_lt(X_T2026, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2028 = cond(X_T2027, X_T2, X_T2026)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1440):(70560, 10080, 1440, 1):275.625 KiB
// Computed true ops: 282240
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
__kernel void kernel_c108_sdk_701(__global float* restrict  X_T2028, __global const float* restrict  X_T2020, __global const float* restrict  X_T2024, __global const float* restrict  X_I_781)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 1408));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((10080 * i2_gid) + (1440 * i3_tid)) + (i4_gid + i4));
        float LX_T2020 = X_T2020[gout_idx];
        float LX_T2024 = X_T2024[(i4_gid + i4)];
        float LX_I_781 = X_I_781[(i4_gid + i4)];
        float LX_T2025 = (LX_T2020 / LX_T2024);
        float LX_T2026 = (LX_T2025 + LX_I_781);
        int LX_T2027 = (LX_T2026 < 0.0f);
        float LX_T2028 = select((float)LX_T2026, (float)0.0f, (int)LX_T2027);
        X_T2028[gout_idx] = LX_T2028;
      }
    }
  }
}
