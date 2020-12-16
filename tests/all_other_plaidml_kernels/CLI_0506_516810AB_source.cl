#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 832 }
// Out stride: { 40768, 5824, 832, 1 }
// Elementwise input X_T1518 shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Elementwise input X_T1541 shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Elementwise input X_I_593 shape: fp32(832):(1):3.25 KiB
// Elementwise input X_I_592 shape: fp32(832):(1):3.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1542 = add(X_T1518, X_T1541)
// Elementwise op: [[pid(Sub)]] X_T1544 = sub(X_T1542, X_I_593)
// Elementwise op: [[pid(Mul)]] X_T1545 = mul(X_T1544, X_I_592)
// Tile size: { 1, 1, 1, 832 }
// Contraction output var shape: fp32(1, 7, 7, 832):(40768, 5824, 832, 1):159.25 KiB
// Computed true ops: 122304
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 416
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_527(__global float* restrict  X_T1542, __global float* restrict  X_T1545, __global const float* restrict  X_T1518, __global const float* restrict  X_T1541, __global const float* restrict  X_I_593, __global const float* restrict  X_I_592)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((5824 * i2_gid) + (832 * i3_gid)) + i4);
      float LX_T1518 = X_T1518[gout_idx];
      float LX_T1541 = X_T1541[gout_idx];
      float LX_I_593 = X_I_593[i4];
      float LX_I_592 = X_I_592[i4];
      float LX_T1542 = (LX_T1518 + LX_T1541);
      float LX_T1544 = (LX_T1542 - LX_I_593);
      float LX_T1545 = (LX_T1544 * LX_I_592);
      X_T1542[gout_idx] = LX_T1542;
      X_T1545[gout_idx] = LX_T1545;
    }
  }
}
