#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 352 }
// Out stride: { 68992, 4928, 352, 1 }
// Elementwise input X_T630 shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Elementwise input X_T634 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_240 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T635 = div(X_T630, X_T634)
// Elementwise op: [[pid(Add, Switch)]] X_T636 = add(X_T635, X_I_240)
// Elementwise op: X_T637 = cmp_lt(X_T636, X_T2)
// Elementwise op: [[pid(Relu)]] X_T638 = cond(X_T637, X_T2, X_T636)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Computed true ops: 275968
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c68_sdk_209(__global float* restrict  X_T638, __global const float* restrict  X_T630, __global const float* restrict  X_T634, __global const float* restrict  X_I_240)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((4928 * (i2_gid + i2_tid)) + (352 * i3)) + (i4_gid + i4_tid));
      float LX_T630 = X_T630[gout_idx];
      float LX_T634 = X_T634[(i4_gid + i4_tid)];
      float LX_I_240 = X_I_240[(i4_gid + i4_tid)];
      float LX_T635 = (LX_T630 / LX_T634);
      float LX_T636 = (LX_T635 + LX_I_240);
      int LX_T637 = (LX_T636 < 0.0f);
      float LX_T638 = select((float)LX_T636, (float)0.0f, (int)LX_T637);
      X_T638[gout_idx] = LX_T638;
    }
  }
}
