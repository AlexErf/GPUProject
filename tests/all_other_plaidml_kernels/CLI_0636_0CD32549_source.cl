#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1152 }
// Out stride: { 56448, 8064, 1152, 1 }
// Elementwise input X_T1976 shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Elementwise input X_T1999 shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Elementwise input X_I_773 shape: fp32(1152):(1):4.5 KiB
// Elementwise input X_I_772 shape: fp32(1152):(1):4.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2000 = add(X_T1976, X_T1999)
// Elementwise op: [[pid(Sub)]] X_T2002 = sub(X_T2000, X_I_773)
// Elementwise op: [[pid(Mul)]] X_T2003 = mul(X_T2002, X_I_772)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Computed true ops: 169344
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c124_sdk_689(__global float* restrict  X_T2000, __global float* restrict  X_T2003, __global const float* restrict  X_T1976, __global const float* restrict  X_T1999, __global const float* restrict  X_I_773, __global const float* restrict  X_I_772)
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
      int gout_idx = (((8064 * i2_gid) + (1152 * i3_tid)) + (i4_gid + i4));
      float LX_T1976 = X_T1976[gout_idx];
      float LX_T1999 = X_T1999[gout_idx];
      float LX_I_773 = X_I_773[(i4_gid + i4)];
      float LX_I_772 = X_I_772[(i4_gid + i4)];
      float LX_T2000 = (LX_T1976 + LX_T1999);
      float LX_T2002 = (LX_T2000 - LX_I_773);
      float LX_T2003 = (LX_T2002 * LX_I_772);
      X_T2000[gout_idx] = LX_T2000;
      X_T2003[gout_idx] = LX_T2003;
    }
  }
}
