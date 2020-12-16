#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 88 }
// Out stride: { 68992, 2464, 88, 1 }
// Elementwise input X_T1222 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise input X_T1226 shape: fp32(88):(1):352 bytes
// Elementwise input X_I_460 shape: fp32(88):(1):352 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T1227 = div(X_T1222, X_T1226)
// Elementwise op: [[pid(Add, Switch)]] X_T1228 = add(X_T1227, X_I_460)
// Elementwise op: X_T1229 = cmp_lt(X_T1228, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1230 = cond(X_T1229, X_T1, X_T1228)
// Tile size: { 1, 28, 1, 88 }
// Contraction output var shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Computed true ops: 275968
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 1008
// Computed mem write: 10752
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_461(__global float* restrict  X_T1230, __global const float* restrict  X_T1222, __global const float* restrict  X_T1226, __global const float* restrict  X_I_460)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 24));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
      {
        int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
        if (i2_cond)
        {
          int i2 = ((8 * i2_lid) + i2_tid);
          int gout_idx = (((2464 * i2) + (88 * i3_gid)) + i4);
          float LX_T1222 = X_T1222[gout_idx];
          float LX_T1226 = X_T1226[i4];
          float LX_I_460 = X_I_460[i4];
          float LX_T1227 = (LX_T1222 / LX_T1226);
          float LX_T1228 = (LX_T1227 + LX_I_460);
          int LX_T1229 = (LX_T1228 < 0.0f);
          float LX_T1230 = select((float)LX_T1228, (float)0.0f, (int)LX_T1229);
          X_T1230[gout_idx] = LX_T1230;
        }
      }
    }
  }
}
