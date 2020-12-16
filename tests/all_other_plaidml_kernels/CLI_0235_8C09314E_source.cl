#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 320 }
// Out stride: { 62720, 4480, 320, 1 }
// Elementwise input X_T578 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_T601 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_I_232 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_231 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T602 = add(X_T578, X_T601)
// Elementwise op: [[pid(Sub)]] X_T604 = sub(X_T602, X_I_232)
// Elementwise op: [[pid(Mul)]] X_T605 = mul(X_T604, X_I_231)
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
__kernel void kernel_c68_sdk_197(__global float* restrict  X_T602, __global float* restrict  X_T605, __global const float* restrict  X_T578, __global const float* restrict  X_T601, __global const float* restrict  X_I_232, __global const float* restrict  X_I_231)
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
      float LX_T578 = X_T578[gout_idx];
      float LX_T601 = X_T601[gout_idx];
      float LX_I_232 = X_I_232[(i4_gid + i4_tid)];
      float LX_I_231 = X_I_231[(i4_gid + i4_tid)];
      float LX_T602 = (LX_T578 + LX_T601);
      float LX_T604 = (LX_T602 - LX_I_232);
      float LX_T605 = (LX_T604 * LX_I_231);
      X_T602[gout_idx] = LX_T602;
      X_T605[gout_idx] = LX_T605;
    }
  }
}
