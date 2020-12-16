#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28416 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 111, 111, 32 }
// Out stride: { 394272, 3552, 32, 1 }
// Elementwise input X_T36 shape: fp32(1, 111, 111, 32):(394272, 3552, 32, 1):1540.12 KiB
// Elementwise input X_T41 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_42 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T42 = div(X_T36, X_T41)
// Elementwise op: [[pid(Add, Switch)]] X_T43 = add(X_T42, X_I_42)
// Elementwise op: X_T44 = cmp_lt(X_T43, X_T1)
// Elementwise op: [[pid(Relu)]] X_T45 = cond(X_T44, X_T1, X_T43)
// Tile size: { 1, 1, 111, 32 }
// Contraction output var shape: fp32(1, 111, 111, 32):(394272, 3552, 32, 1):1540.12 KiB
// Computed true ops: 1577088
// Computed work groups: 111
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1332
// Computed mem write: 14208
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28416, 1, 1
__kernel void kernel_c42_sdk_2(__global float* restrict  X_T45, __global const float* restrict  X_T36, __global const float* restrict  X_T41, __global const float* restrict  X_I_42)
{
  int tid = get_local_id(0);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i3_lid = 0; i3_lid < 14; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 13) || (i3_tid < 7));
    if (i3_cond)
    {
      int i3 = ((8 * i3_lid) + i3_tid);
      int gout_idx = (((3552 * i2_gid) + (32 * i3)) + i4_tid);
      float LX_T36 = X_T36[gout_idx];
      float LX_T41 = X_T41[i4_tid];
      float LX_I_42 = X_I_42[i4_tid];
      float LX_T42 = (LX_T36 / LX_T41);
      float LX_T43 = (LX_T42 + LX_I_42);
      int LX_T44 = (LX_T43 < 0.0f);
      float LX_T45 = select((float)LX_T43, (float)0.0f, (int)LX_T44);
      X_T45[gout_idx] = LX_T45;
    }
  }
}
