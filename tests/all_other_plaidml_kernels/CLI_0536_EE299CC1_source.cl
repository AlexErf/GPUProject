#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1024 }
// Out stride: { 50176, 7168, 1024, 1 }
// Elementwise input X_T1668 shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Elementwise input X_T1691 shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Elementwise input X_I_653 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_652 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1692 = add(X_T1668, X_T1691)
// Elementwise op: [[pid(Sub)]] X_T1694 = sub(X_T1692, X_I_653)
// Elementwise op: [[pid(Mul)]] X_T1695 = mul(X_T1694, X_I_652)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Computed true ops: 150528
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c108_sdk_581(__global float* restrict  X_T1692, __global float* restrict  X_T1695, __global const float* restrict  X_T1668, __global const float* restrict  X_T1691, __global const float* restrict  X_I_653, __global const float* restrict  X_I_652)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((7168 * i2_gid) + (1024 * i3_tid)) + (i4_gid + i4));
      float LX_T1668 = X_T1668[gout_idx];
      float LX_T1691 = X_T1691[gout_idx];
      float LX_I_653 = X_I_653[(i4_gid + i4)];
      float LX_I_652 = X_I_652[(i4_gid + i4)];
      float LX_T1692 = (LX_T1668 + LX_T1691);
      float LX_T1694 = (LX_T1692 - LX_I_653);
      float LX_T1695 = (LX_T1694 * LX_I_652);
      X_T1692[gout_idx] = LX_T1692;
      X_T1695[gout_idx] = LX_T1695;
    }
  }
}
