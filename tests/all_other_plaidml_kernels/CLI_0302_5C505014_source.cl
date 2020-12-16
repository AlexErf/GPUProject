#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 352 }
// Out stride: { 68992, 4928, 352, 1 }
// Elementwise input X_T650 shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Elementwise input X_T654 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_240 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T655 = div(X_T650, X_T654)
// Elementwise op: [[pid(Add, Switch)]] X_T656 = add(X_T655, X_I_240)
// Elementwise op: X_T657 = cmp_lt(X_T656, X_T2)
// Elementwise op: [[pid(Relu)]] X_T658 = cond(X_T657, X_T2, X_T656)
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
__kernel void kernel_c108_sdk_209(__global float* restrict  X_T658, __global const float* restrict  X_T650, __global const float* restrict  X_T654, __global const float* restrict  X_I_240)
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
      float LX_T650 = X_T650[gout_idx];
      float LX_T654 = X_T654[(i4_gid + i4_tid)];
      float LX_I_240 = X_I_240[(i4_gid + i4_tid)];
      float LX_T655 = (LX_T650 / LX_T654);
      float LX_T656 = (LX_T655 + LX_I_240);
      int LX_T657 = (LX_T656 < 0.0f);
      float LX_T658 = select((float)LX_T656, (float)0.0f, (int)LX_T657);
      X_T658[gout_idx] = LX_T658;
    }
  }
}
