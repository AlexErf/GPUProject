#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 96 }
// Out stride: { 27744, 1632, 96, 1 }
// Elementwise input X_T370 shape: fp32(1, 17, 17, 96):(27744, 1632, 96, 1):108.375 KiB
// Elementwise input X_T374 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_135 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T375 = div(X_T370, X_T374)
// Elementwise op: [[pid(Add, Switch)]] X_T376 = add(X_T375, X_I_135)
// Elementwise op: X_T377 = cmp_lt(X_T376, X_T2)
// Elementwise op: [[pid(Relu)]] X_T378 = cond(X_T377, X_T2, X_T376)
// Tile size: { 1, 2, 2, 96 }
// Contraction output var shape: fp32(1, 17, 17, 96):(27744, 1632, 96, 1):108.375 KiB
// Computed true ops: 110976
// Computed work groups: 81
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 9, 1
__kernel void kernel_c56_sdk_122(__global float* restrict  X_T378, __global const float* restrict  X_T370, __global const float* restrict  X_T374, __global const float* restrict  X_I_135)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 16) || (i3_tid < 1));
  if (i3_cond)
  {
    int i2_cond = ((i2_gid != 16) || (i2_tid < 1));
    if (i2_cond)
    {
      for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
      {
        int i4_cond = ((i4_lid < 1) || (i4_tid < 32));
        if (i4_cond)
        {
          int i4 = ((64 * i4_lid) + i4_tid);
          int gout_idx = (((1632 * (i2_gid + i2_tid)) + (96 * (i3_gid + i3_tid))) + i4);
          float LX_T370 = X_T370[gout_idx];
          float LX_T374 = X_T374[i4];
          float LX_I_135 = X_I_135[i4];
          float LX_T375 = (LX_T370 / LX_T374);
          float LX_T376 = (LX_T375 + LX_I_135);
          int LX_T377 = (LX_T376 < 0.0f);
          float LX_T378 = select((float)LX_T376, (float)0.0f, (int)LX_T377);
          X_T378[gout_idx] = LX_T378;
        }
      }
    }
  }
}
