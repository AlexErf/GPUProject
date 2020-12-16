#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 160 }
// Out stride: { 501760, 8960, 160, 1 }
// Elementwise input X_T159 shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Elementwise input X_T163 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_58 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T164 = div(X_T159, X_T163)
// Elementwise op: [[pid(Add, Switch)]] X_T165 = add(X_T164, X_I_58)
// Elementwise op: X_T166 = cmp_lt(X_T165, X_T2)
// Elementwise op: [[pid(Relu)]] X_T167 = cond(X_T166, X_T2, X_T165)
// Tile size: { 1, 4, 4, 160 }
// Contraction output var shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Computed true ops: 2007040
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 960
// Computed mem write: 10240
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c108_sdk_35(__global float* restrict  X_T167, __global const float* restrict  X_T159, __global const float* restrict  X_T163, __global const float* restrict  X_I_58)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((8960 * (i2_gid + i2)) + (160 * (i3_gid + i3_tid))) + i4);
      float LX_T159 = X_T159[gout_idx];
      float LX_T163 = X_T163[i4];
      float LX_I_58 = X_I_58[i4];
      float LX_T164 = (LX_T159 / LX_T163);
      float LX_T165 = (LX_T164 + LX_I_58);
      int LX_T166 = (LX_T165 < 0.0f);
      float LX_T167 = select((float)LX_T165, (float)0.0f, (int)LX_T166);
      X_T167[gout_idx] = LX_T167;
    }
  }
}
