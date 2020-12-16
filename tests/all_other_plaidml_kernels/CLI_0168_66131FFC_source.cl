#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 44 }
// Out stride: { 34496, 1232, 44, 1 }
// Elementwise input X_T679 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise input X_T683 shape: fp32(44):(1):176 bytes
// Elementwise input X_I_250 shape: fp32(44):(1):176 bytes
// Elementwise input X_T521 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T684 = div(X_T679, X_T683)
// Elementwise op: [[pid(Add, Switch)]] X_T685 = add(X_T684, X_I_250)
// Elementwise op: [[pid(Add)]] X_T686 = add(X_T685, X_T521)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 103488
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c42_sdk_248(__global float* restrict  X_T686, __global const float* restrict  X_T679, __global const float* restrict  X_T683, __global const float* restrict  X_I_250, __global const float* restrict  X_T521)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = ((i4_gid != 32) || (i4_tid < 12));
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1232 * i2) + (44 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
      float LX_T679 = X_T679[gout_idx];
      float LX_T683 = X_T683[(i4_gid + i4_tid)];
      float LX_I_250 = X_I_250[(i4_gid + i4_tid)];
      float LX_T521 = X_T521[gout_idx];
      float LX_T684 = (LX_T679 / LX_T683);
      float LX_T685 = (LX_T684 + LX_I_250);
      float LX_T686 = (LX_T685 + LX_T521);
      X_T686[gout_idx] = LX_T686;
    }
  }
}
