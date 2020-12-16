#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 32 }
// Out stride: { 39200, 1120, 32, 1 }
// Elementwise input X_T165 shape: fp32(1, 35, 35, 32):(39200, 1120, 32, 1):153.125 KiB
// Elementwise input X_T169 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_80 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T170 = div(X_T165, X_T169)
// Elementwise op: [[pid(Add, Switch)]] X_T171 = add(X_T170, X_I_80)
// Elementwise op: X_T172 = cmp_lt(X_T171, X_T2)
// Elementwise op: [[pid(Relu)]] X_T173 = cond(X_T172, X_T2, X_T171)
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
__kernel void kernel_c56_sdk_45(__global float* restrict  X_T173, __global const float* restrict  X_T165, __global const float* restrict  X_T169, __global const float* restrict  X_I_80)
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
      float LX_T165 = X_T165[gout_idx];
      float LX_T169 = X_T169[i4_tid];
      float LX_I_80 = X_I_80[i4_tid];
      float LX_T170 = (LX_T165 / LX_T169);
      float LX_T171 = (LX_T170 + LX_I_80);
      int LX_T172 = (LX_T171 < 0.0f);
      float LX_T173 = select((float)LX_T171, (float)0.0f, (int)LX_T172);
      X_T173[gout_idx] = LX_T173;
    }
  }
}
