#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 11 }
// Out stride: { 34496, 616, 11, 1 }
// Elementwise input X_T87 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise input X_T91 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_49 shape: fp32(11):(1):44 bytes
// Elementwise input X_T57 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T92 = div(X_T87, X_T91)
// Elementwise op: [[pid(Add, Switch)]] X_T93 = add(X_T92, X_I_49)
// Elementwise op: [[pid(Add)]] X_T94 = add(X_T57, X_T93)
// Tile size: { 1, 56, 2, 11 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 103488
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 1792
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_18(__global float* restrict  X_T94, __global const float* restrict  X_T87, __global const float* restrict  X_T91, __global const float* restrict  X_I_49, __global const float* restrict  X_T57)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 16);
  int i3_tid = ((tid / 16) % 2);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 11);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((616 * i2) + (11 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T87 = X_T87[gout_idx];
      float LX_T91 = X_T91[i4_tid];
      float LX_I_49 = X_I_49[i4_tid];
      float LX_T57 = X_T57[gout_idx];
      float LX_T92 = (LX_T87 / LX_T91);
      float LX_T93 = (LX_T92 + LX_I_49);
      float LX_T94 = (LX_T57 + LX_T93);
      X_T94[gout_idx] = LX_T94;
    }
  }
}
