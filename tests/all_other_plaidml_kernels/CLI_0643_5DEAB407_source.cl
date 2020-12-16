#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1184 }
// Out stride: { 58016, 8288, 1184, 1 }
// Elementwise input X_T2028 shape: fp32(1, 7, 7, 1184):(58016, 8288, 1184, 1):226.625 KiB
// Elementwise input X_T2032 shape: fp32(1184):(1):4.625 KiB
// Elementwise input X_I_781 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2033 = div(X_T2028, X_T2032)
// Elementwise op: [[pid(Add, Switch)]] X_T2034 = add(X_T2033, X_I_781)
// Elementwise op: X_T2035 = cmp_lt(X_T2034, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2036 = cond(X_T2035, X_T2, X_T2034)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1184):(58016, 8288, 1184, 1):226.625 KiB
// Computed true ops: 232064
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
__kernel void kernel_c124_sdk_701(__global float* restrict  X_T2036, __global const float* restrict  X_T2028, __global const float* restrict  X_T2032, __global const float* restrict  X_I_781)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 1152));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((8288 * i2_gid) + (1184 * i3_tid)) + (i4_gid + i4));
        float LX_T2028 = X_T2028[gout_idx];
        float LX_T2032 = X_T2032[(i4_gid + i4)];
        float LX_I_781 = X_I_781[(i4_gid + i4)];
        float LX_T2033 = (LX_T2028 / LX_T2032);
        float LX_T2034 = (LX_T2033 + LX_I_781);
        int LX_T2035 = (LX_T2034 < 0.0f);
        float LX_T2036 = select((float)LX_T2034, (float)0.0f, (int)LX_T2035);
        X_T2036[gout_idx] = LX_T2036;
      }
    }
  }
}
