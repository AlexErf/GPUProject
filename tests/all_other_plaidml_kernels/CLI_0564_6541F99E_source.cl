#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 52 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1664 }
// Out stride: { 326144, 23296, 1664, 1 }
// Elementwise input X_T1656 shape: fp32(1, 14, 14, 1664):(326144, 23296, 1664, 1):1274 KiB
// Elementwise input X_T1679 shape: fp32(1, 14, 14, 1664):(326144, 23296, 1664, 1):1274 KiB
// Elementwise input X_I_652 shape: fp32(1664):(1):6.5 KiB
// Elementwise input X_I_651 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1680 = add(X_T1656, X_T1679)
// Elementwise op: [[pid(Sub)]] X_T1682 = sub(X_T1680, X_I_652)
// Elementwise op: [[pid(Mul)]] X_T1683 = mul(X_T1682, X_I_651)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1664):(326144, 23296, 1664, 1):1274 KiB
// Computed true ops: 978432
// Computed work groups: 364
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 52, 1
__kernel void kernel_c124_sdk_575(__global float* restrict  X_T1680, __global float* restrict  X_T1683, __global const float* restrict  X_T1656, __global const float* restrict  X_T1679, __global const float* restrict  X_I_652, __global const float* restrict  X_I_651)
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
      int gout_idx = (((23296 * (i2_gid + i2_tid)) + (1664 * i3)) + (i4_gid + i4_tid));
      float LX_T1656 = X_T1656[gout_idx];
      float LX_T1679 = X_T1679[gout_idx];
      float LX_I_652 = X_I_652[(i4_gid + i4_tid)];
      float LX_I_651 = X_I_651[(i4_gid + i4_tid)];
      float LX_T1680 = (LX_T1656 + LX_T1679);
      float LX_T1682 = (LX_T1680 - LX_I_652);
      float LX_T1683 = (LX_T1682 * LX_I_651);
      X_T1680[gout_idx] = LX_T1680;
      X_T1683[gout_idx] = LX_T1683;
    }
  }
}
