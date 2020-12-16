#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 19 19
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 37, 37, 728 }
// Out stride: { 996632, 26936, 728, 1 }
// Elementwise input X_T211 shape: fp32(1, 37, 37, 728):(996632, 26936, 728, 1):3893.09 KiB
// Elementwise input X_T215 shape: fp32(728):(1):2.84375 KiB
// Elementwise input X_I_147 shape: fp32(728):(1):2.84375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T216 = div(X_T211, X_T215)
// Elementwise op: [[pid(Add, Switch)]] X_T217 = add(X_T216, X_I_147)
// Elementwise op: X_T218 = cmp_lt(X_T217, X_T2)
// Elementwise op: [[pid(Relu)]] X_T219 = cond(X_T218, X_T2, X_T217)
// Tile size: { 1, 2, 2, 256 }
// Contraction output var shape: fp32(1, 37, 37, 728):(996632, 26936, 728, 1):3893.09 KiB
// Computed true ops: 3986528
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
__kernel void kernel_c28_sdk_67(__global float* restrict  X_T219, __global const float* restrict  X_T211, __global const float* restrict  X_T215, __global const float* restrict  X_I_147)
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
          float LX_T211 = X_T211[gout_idx];
          float LX_T215 = X_T215[(i4_gid + i4)];
          float LX_I_147 = X_I_147[(i4_gid + i4)];
          float LX_T216 = (LX_T211 / LX_T215);
          float LX_T217 = (LX_T216 + LX_I_147);
          int LX_T218 = (LX_T217 < 0.0f);
          float LX_T219 = select((float)LX_T217, (float)0.0f, (int)LX_T218);
          X_T219[gout_idx] = LX_T219;
        }
      }
    }
  }
}
