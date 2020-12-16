#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 160 }
// Out stride: { 46240, 2720, 160, 1 }
// Elementwise input X_T961 shape: fp32(1, 17, 17, 160):(46240, 2720, 160, 1):180.625 KiB
// Elementwise input X_T965 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_345 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T966 = div(X_T961, X_T965)
// Elementwise op: [[pid(Add, Switch)]] X_T967 = add(X_T966, X_I_345)
// Elementwise op: X_T968 = cmp_lt(X_T967, X_T2)
// Elementwise op: [[pid(Relu)]] X_T969 = cond(X_T968, X_T2, X_T967)
// Tile size: { 1, 2, 17, 32 }
// Contraction output var shape: fp32(1, 17, 17, 160):(46240, 2720, 160, 1):180.625 KiB
// Computed true ops: 184960
// Computed work groups: 45
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 408
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 9, 1
__kernel void kernel_c51_sdk_315(__global float* restrict  X_T969, __global const float* restrict  X_T961, __global const float* restrict  X_T965, __global const float* restrict  X_I_345)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i2_cond = ((i2_gid != 16) || (i2_tid < 1));
  if (i2_cond)
  {
    for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
    {
      int i3_cond = ((i3_lid < 4) || (i3_tid < 1));
      if (i3_cond)
      {
        int i3 = ((4 * i3_lid) + i3_tid);
        int gout_idx = (((2720 * (i2_gid + i2_tid)) + (160 * i3)) + (i4_gid + i4_tid));
        float LX_T961 = X_T961[gout_idx];
        float LX_T965 = X_T965[(i4_gid + i4_tid)];
        float LX_I_345 = X_I_345[(i4_gid + i4_tid)];
        float LX_T966 = (LX_T961 / LX_T965);
        float LX_T967 = (LX_T966 + LX_I_345);
        int LX_T968 = (LX_T967 < 0.0f);
        float LX_T969 = select((float)LX_T967, (float)0.0f, (int)LX_T968);
        X_T969[gout_idx] = LX_T969;
      }
    }
  }
}
