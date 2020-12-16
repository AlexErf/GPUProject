#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 88 }
// Out stride: { 68992, 2464, 88, 1 }
// Elementwise input X_T1203 shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Elementwise input X_T1207 shape: fp32(88):(1):352 bytes
// Elementwise input X_I_22 shape: fp32(88):(1):352 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T1208 = div(X_T1203, X_T1207)
// Elementwise op: [[pid(Add, Switch)]] X_T1209 = add(X_T1208, X_I_22)
// Elementwise op: X_T1304 = cmp_lt(X_T1209, X_T1)
// Elementwise op: [[pid(Relu)]] X_T1305 = cond(X_T1304, X_T1, X_T1209)
// Tile size: { 1, 28, 1, 88 }
// Contraction output var shape: fp32(1, 28, 28, 88):(68992, 2464, 88, 1):269.5 KiB
// Computed true ops: 275968
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 1008
// Computed mem write: 21504
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_454(__global float* restrict  X_T1209, __global float* restrict  X_T1305, __global const float* restrict  X_T1203, __global const float* restrict  X_T1207, __global const float* restrict  X_I_22)
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
          float LX_T1203 = X_T1203[gout_idx];
          float LX_T1207 = X_T1207[i4];
          float LX_I_22 = X_I_22[i4];
          float LX_T1208 = (LX_T1203 / LX_T1207);
          float LX_T1209 = (LX_T1208 + LX_I_22);
          int LX_T1304 = (LX_T1209 < 0.0f);
          float LX_T1305 = select((float)LX_T1209, (float)0.0f, (int)LX_T1304);
          X_T1209[gout_idx] = LX_T1209;
          X_T1305[gout_idx] = LX_T1305;
        }
      }
    }
  }
}
