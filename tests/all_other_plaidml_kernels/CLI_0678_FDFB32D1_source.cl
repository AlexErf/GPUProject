#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1408 }
// Out stride: { 68992, 9856, 1408, 1 }
// Elementwise input X_T2203 shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Elementwise input X_T2207 shape: fp32(1408):(1):5.5 KiB
// Elementwise input X_I_851 shape: fp32(1408):(1):5.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2208 = div(X_T2203, X_T2207)
// Elementwise op: [[pid(Add, Switch)]] X_T2209 = add(X_T2208, X_I_851)
// Elementwise op: X_T2210 = cmp_lt(X_T2209, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2211 = cond(X_T2210, X_T2, X_T2209)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1408):(68992, 9856, 1408, 1):269.5 KiB
// Computed true ops: 275968
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
__kernel void kernel_c124_sdk_764(__global float* restrict  X_T2211, __global const float* restrict  X_T2203, __global const float* restrict  X_T2207, __global const float* restrict  X_I_851)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((9856 * i2_gid) + (1408 * i3_tid)) + (i4_gid + i4));
      float LX_T2203 = X_T2203[gout_idx];
      float LX_T2207 = X_T2207[(i4_gid + i4)];
      float LX_I_851 = X_I_851[(i4_gid + i4)];
      float LX_T2208 = (LX_T2203 / LX_T2207);
      float LX_T2209 = (LX_T2208 + LX_I_851);
      int LX_T2210 = (LX_T2209 < 0.0f);
      float LX_T2211 = select((float)LX_T2209, (float)0.0f, (int)LX_T2210);
      X_T2211[gout_idx] = LX_T2211;
    }
  }
}
