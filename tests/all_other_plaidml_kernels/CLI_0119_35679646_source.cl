#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 32 }
// Out stride: { 39200, 1120, 32, 1 }
// Elementwise input X_T178 shape: fp32(1, 35, 35, 32):(39200, 1120, 32, 1):153.125 KiB
// Elementwise input X_T182 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_65 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T183 = div(X_T178, X_T182)
// Elementwise op: [[pid(Add, Switch)]] X_T184 = add(X_T183, X_I_65)
// Elementwise op: X_T185 = cmp_lt(X_T184, X_T2)
// Elementwise op: [[pid(Relu)]] X_T186 = cond(X_T185, X_T2, X_T184)
// Tile size: { 1, 35, 1, 32 }
// Contraction output var shape: fp32(1, 35, 35, 32):(39200, 1120, 32, 1):153.125 KiB
// Computed true ops: 156800
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 420
// Computed mem write: 4480
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c51_sdk_50(__global float* restrict  X_T186, __global const float* restrict  X_T178, __global const float* restrict  X_T182, __global const float* restrict  X_I_65)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 5; i2_lid += 1)
  {
    int i2_cond = ((i2_lid < 4) || (i2_tid < 3));
    if (i2_cond)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((1120 * i2) + (32 * i3_gid)) + i4_tid);
      float LX_T178 = X_T178[gout_idx];
      float LX_T182 = X_T182[i4_tid];
      float LX_I_65 = X_I_65[i4_tid];
      float LX_T183 = (LX_T178 / LX_T182);
      float LX_T184 = (LX_T183 + LX_I_65);
      int LX_T185 = (LX_T184 < 0.0f);
      float LX_T186 = select((float)LX_T184, (float)0.0f, (int)LX_T185);
      X_T186[gout_idx] = LX_T186;
    }
  }
}
