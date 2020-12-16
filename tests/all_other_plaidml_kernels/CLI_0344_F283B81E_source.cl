#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 480 }
// Out stride: { 94080, 6720, 480, 1 }
// Elementwise input X_T731 shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Elementwise input X_T754 shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Elementwise input X_I_282 shape: fp32(480):(1):1.875 KiB
// Elementwise input X_I_281 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T755 = add(X_T731, X_T754)
// Elementwise op: [[pid(Sub)]] X_T757 = sub(X_T755, X_I_282)
// Elementwise op: [[pid(Mul)]] X_T758 = mul(X_T757, X_I_281)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Computed true ops: 282240
// Computed work groups: 105
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 15, 1
__kernel void kernel_c124_sdk_242(__global float* restrict  X_T755, __global float* restrict  X_T758, __global const float* restrict  X_T731, __global const float* restrict  X_T754, __global const float* restrict  X_I_282, __global const float* restrict  X_I_281)
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
      int gout_idx = (((6720 * (i2_gid + i2_tid)) + (480 * i3)) + (i4_gid + i4_tid));
      float LX_T731 = X_T731[gout_idx];
      float LX_T754 = X_T754[gout_idx];
      float LX_I_282 = X_I_282[(i4_gid + i4_tid)];
      float LX_I_281 = X_I_281[(i4_gid + i4_tid)];
      float LX_T755 = (LX_T731 + LX_T754);
      float LX_T757 = (LX_T755 - LX_I_282);
      float LX_T758 = (LX_T757 * LX_I_281);
      X_T755[gout_idx] = LX_T755;
      X_T758[gout_idx] = LX_T758;
    }
  }
}
