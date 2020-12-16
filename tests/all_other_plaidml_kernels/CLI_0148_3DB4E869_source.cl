#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 320 }
// Out stride: { 15680, 2240, 320, 1 }
// Elementwise input X_T698 shape: fp32(1, 7, 7, 320):(15680, 2240, 320, 1):61.25 KiB
// Elementwise input X_T702 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_6 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T703 = div(X_T698, X_T702)
// Elementwise op: [[pid(Add, Switch)]] X_T704 = add(X_T703, X_I_6)
// Tile size: { 1, 7, 1, 64 }
// Contraction output var shape: fp32(1, 7, 7, 320):(15680, 2240, 320, 1):61.25 KiB
// Computed true ops: 31360
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 168
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 7, 1
__kernel void kernel_c43_sdk_191(__global float* restrict  X_T704, __global const float* restrict  X_T698, __global const float* restrict  X_T702, __global const float* restrict  X_I_6)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  int i2_cond = (i2_tid < 7);
  if (i2_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((2240 * i2_tid) + (320 * i3_gid)) + (i4_gid + i4));
      float LX_T698 = X_T698[gout_idx];
      float LX_T702 = X_T702[(i4_gid + i4)];
      float LX_I_6 = X_I_6[(i4_gid + i4)];
      float LX_T703 = (LX_T698 / LX_T702);
      float LX_T704 = (LX_T703 + LX_I_6);
      X_T704[gout_idx] = LX_T704;
    }
  }
}
