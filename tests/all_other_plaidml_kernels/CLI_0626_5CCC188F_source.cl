#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1536 }
// Out stride: { 75264, 10752, 1536, 1 }
// Elementwise input X_T2095 shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Elementwise input X_T2099 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_811 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2100 = div(X_T2095, X_T2099)
// Elementwise op: [[pid(Add, Switch)]] X_T2101 = add(X_T2100, X_I_811)
// Elementwise op: X_T2102 = cmp_lt(X_T2101, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2103 = cond(X_T2102, X_T2, X_T2101)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Computed true ops: 301056
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
__kernel void kernel_c108_sdk_728(__global float* restrict  X_T2103, __global const float* restrict  X_T2095, __global const float* restrict  X_T2099, __global const float* restrict  X_I_811)
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
      int gout_idx = (((10752 * i2_gid) + (1536 * i3_tid)) + (i4_gid + i4));
      float LX_T2095 = X_T2095[gout_idx];
      float LX_T2099 = X_T2099[(i4_gid + i4)];
      float LX_I_811 = X_I_811[(i4_gid + i4)];
      float LX_T2100 = (LX_T2095 / LX_T2099);
      float LX_T2101 = (LX_T2100 + LX_I_811);
      int LX_T2102 = (LX_T2101 < 0.0f);
      float LX_T2103 = select((float)LX_T2101, (float)0.0f, (int)LX_T2102);
      X_T2103[gout_idx] = LX_T2103;
    }
  }
}
