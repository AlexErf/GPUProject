#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 46 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1472 }
// Out stride: { 288512, 20608, 1472, 1 }
// Elementwise input X_T1506 shape: fp32(1, 14, 14, 1472):(288512, 20608, 1472, 1):1127 KiB
// Elementwise input X_T1529 shape: fp32(1, 14, 14, 1472):(288512, 20608, 1472, 1):1127 KiB
// Elementwise input X_I_592 shape: fp32(1472):(1):5.75 KiB
// Elementwise input X_I_591 shape: fp32(1472):(1):5.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1530 = add(X_T1506, X_T1529)
// Elementwise op: [[pid(Sub)]] X_T1532 = sub(X_T1530, X_I_592)
// Elementwise op: [[pid(Mul)]] X_T1533 = mul(X_T1532, X_I_591)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1472):(288512, 20608, 1472, 1):1127 KiB
// Computed true ops: 865536
// Computed work groups: 322
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 46, 1
__kernel void kernel_c124_sdk_521(__global float* restrict  X_T1530, __global float* restrict  X_T1533, __global const float* restrict  X_T1506, __global const float* restrict  X_T1529, __global const float* restrict  X_I_592, __global const float* restrict  X_I_591)
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
      int gout_idx = (((20608 * (i2_gid + i2_tid)) + (1472 * i3)) + (i4_gid + i4_tid));
      float LX_T1506 = X_T1506[gout_idx];
      float LX_T1529 = X_T1529[gout_idx];
      float LX_I_592 = X_I_592[(i4_gid + i4_tid)];
      float LX_I_591 = X_I_591[(i4_gid + i4_tid)];
      float LX_T1530 = (LX_T1506 + LX_T1529);
      float LX_T1532 = (LX_T1530 - LX_I_592);
      float LX_T1533 = (LX_T1532 * LX_I_591);
      X_T1530[gout_idx] = LX_T1530;
      X_T1533[gout_idx] = LX_T1533;
    }
  }
}
