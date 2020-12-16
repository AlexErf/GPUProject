#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1152 }
// Out stride: { 56448, 8064, 1152, 1 }
// Elementwise input X_T1768 shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Elementwise input X_T1791 shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Elementwise input X_I_693 shape: fp32(1152):(1):4.5 KiB
// Elementwise input X_I_692 shape: fp32(1152):(1):4.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1792 = add(X_T1768, X_T1791)
// Elementwise op: [[pid(Sub)]] X_T1794 = sub(X_T1792, X_I_693)
// Elementwise op: [[pid(Mul)]] X_T1795 = mul(X_T1794, X_I_692)
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
__kernel void kernel_c108_sdk_617(__global float* restrict  X_T1792, __global float* restrict  X_T1795, __global const float* restrict  X_T1768, __global const float* restrict  X_T1791, __global const float* restrict  X_I_693, __global const float* restrict  X_I_692)
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
      float LX_T1768 = X_T1768[gout_idx];
      float LX_T1791 = X_T1791[gout_idx];
      float LX_I_693 = X_I_693[(i4_gid + i4)];
      float LX_I_692 = X_I_692[(i4_gid + i4)];
      float LX_T1792 = (LX_T1768 + LX_T1791);
      float LX_T1794 = (LX_T1792 - LX_I_693);
      float LX_T1795 = (LX_T1794 * LX_I_692);
      X_T1792[gout_idx] = LX_T1792;
      X_T1795[gout_idx] = LX_T1795;
    }
  }
}
