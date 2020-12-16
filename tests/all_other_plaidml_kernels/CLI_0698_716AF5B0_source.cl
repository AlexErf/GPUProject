#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1536 }
// Out stride: { 75264, 10752, 1536, 1 }
// Elementwise input X_T2303 shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Elementwise input X_T2307 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_891 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2308 = div(X_T2303, X_T2307)
// Elementwise op: [[pid(Add, Switch)]] X_T2309 = add(X_T2308, X_I_891)
// Elementwise op: X_T2310 = cmp_lt(X_T2309, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2311 = cond(X_T2310, X_T2, X_T2309)
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
__kernel void kernel_c124_sdk_800(__global float* restrict  X_T2311, __global const float* restrict  X_T2303, __global const float* restrict  X_T2307, __global const float* restrict  X_I_891)
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
      float LX_T2303 = X_T2303[gout_idx];
      float LX_T2307 = X_T2307[(i4_gid + i4)];
      float LX_I_891 = X_I_891[(i4_gid + i4)];
      float LX_T2308 = (LX_T2303 / LX_T2307);
      float LX_T2309 = (LX_T2308 + LX_I_891);
      int LX_T2310 = (LX_T2309 < 0.0f);
      float LX_T2311 = select((float)LX_T2309, (float)0.0f, (int)LX_T2310);
      X_T2311[gout_idx] = LX_T2311;
    }
  }
}
