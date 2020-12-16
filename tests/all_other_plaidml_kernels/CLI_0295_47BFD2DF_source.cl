#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 320 }
// Out stride: { 62720, 4480, 320, 1 }
// Elementwise input X_T598 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_T621 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_I_232 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_231 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T622 = add(X_T598, X_T621)
// Elementwise op: [[pid(Sub)]] X_T624 = sub(X_T622, X_I_232)
// Elementwise op: [[pid(Mul)]] X_T625 = mul(X_T624, X_I_231)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Computed true ops: 188160
// Computed work groups: 70
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 10, 1
__kernel void kernel_c108_sdk_197(__global float* restrict  X_T622, __global float* restrict  X_T625, __global const float* restrict  X_T598, __global const float* restrict  X_T621, __global const float* restrict  X_I_232, __global const float* restrict  X_I_231)
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
      int gout_idx = (((4480 * (i2_gid + i2_tid)) + (320 * i3)) + (i4_gid + i4_tid));
      float LX_T598 = X_T598[gout_idx];
      float LX_T621 = X_T621[gout_idx];
      float LX_I_232 = X_I_232[(i4_gid + i4_tid)];
      float LX_I_231 = X_I_231[(i4_gid + i4_tid)];
      float LX_T622 = (LX_T598 + LX_T621);
      float LX_T624 = (LX_T622 - LX_I_232);
      float LX_T625 = (LX_T624 * LX_I_231);
      X_T622[gout_idx] = LX_T622;
      X_T625[gout_idx] = LX_T625;
    }
  }
}
