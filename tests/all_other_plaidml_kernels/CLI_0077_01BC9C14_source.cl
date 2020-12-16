#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 19 19
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 37, 37, 728 }
// Out stride: { 996632, 26936, 728, 1 }
// Elementwise input X_T224 shape: fp32(1, 37, 37, 728):(996632, 26936, 728, 1):3893.09 KiB
// Elementwise input X_T228 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_I_142 shape: fp32(728):(1):2.84375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T229 = div(X_T224, X_T228)
// Elementwise op: [[pid(Add, Switch)]] X_T230 = add(X_T229, X_I_142)
// Tile size: { 1, 2, 2, 256 }
// Contraction output var shape: fp32(1, 37, 37, 728):(996632, 26936, 728, 1):3893.09 KiB
// Computed true ops: 1993264
// Computed work groups: 1083
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 384
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 19, 19
__kernel void kernel_c28_sdk_71(__global float* restrict  X_T230, __global const float* restrict  X_T224, __global const float* restrict  X_T228, __global const float* restrict  X_I_142)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 256);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || ((i4_gid != 512) || (i4_tid < 24)));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int i3_cond = ((i3_gid != 36) || (i3_tid < 1));
      if (i3_cond)
      {
        int i2_cond = ((i2_gid != 36) || (i2_tid < 1));
        if (i2_cond)
        {
          int gout_idx = (((26936 * (i2_gid + i2_tid)) + (728 * (i3_gid + i3_tid))) + (i4_gid + i4));
          float LX_T224 = X_T224[gout_idx];
          float LX_T228 = X_T228[(i4_gid + i4)];
          float LX_I_142 = X_I_142[(i4_gid + i4)];
          float LX_T229 = (LX_T224 / LX_T228);
          float LX_T230 = (LX_T229 + LX_I_142);
          X_T230[gout_idx] = LX_T230;
        }
      }
    }
  }
}
