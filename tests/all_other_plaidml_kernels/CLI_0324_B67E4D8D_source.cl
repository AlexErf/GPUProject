#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 352 }
// Out stride: { 68992, 4928, 352, 1 }
// Elementwise input X_T631 shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Elementwise input X_T654 shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Elementwise input X_I_242 shape: fp32(352):(1):1.375 KiB
// Elementwise input X_I_241 shape: fp32(352):(1):1.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T655 = add(X_T631, X_T654)
// Elementwise op: [[pid(Sub)]] X_T657 = sub(X_T655, X_I_242)
// Elementwise op: [[pid(Mul)]] X_T658 = mul(X_T657, X_I_241)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 352):(68992, 4928, 352, 1):269.5 KiB
// Computed true ops: 206976
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c124_sdk_206(__global float* restrict  X_T655, __global float* restrict  X_T658, __global const float* restrict  X_T631, __global const float* restrict  X_T654, __global const float* restrict  X_I_242, __global const float* restrict  X_I_241)
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
      float LX_T631 = X_T631[gout_idx];
      float LX_T654 = X_T654[gout_idx];
      float LX_I_242 = X_I_242[(i4_gid + i4_tid)];
      float LX_I_241 = X_I_241[(i4_gid + i4_tid)];
      float LX_T655 = (LX_T631 + LX_T654);
      float LX_T657 = (LX_T655 - LX_I_242);
      float LX_T658 = (LX_T657 * LX_I_241);
      X_T655[gout_idx] = LX_T655;
      X_T658[gout_idx] = LX_T658;
    }
  }
}
