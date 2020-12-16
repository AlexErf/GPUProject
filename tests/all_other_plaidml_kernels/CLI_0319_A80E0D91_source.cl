#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 320 }
// Out stride: { 62720, 4480, 320, 1 }
// Elementwise input X_T606 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_T629 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_I_232 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_231 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(Concatenate)]] X_T630 = add(X_T606, X_T629)
// Elementwise op: [[pid(Sub)]] X_T632 = sub(X_T630, X_I_232)
// Elementwise op: [[pid(Mul)]] X_T633 = mul(X_T632, X_I_231)
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
__kernel void kernel_c124_sdk_197(__global float* restrict  X_T630, __global float* restrict  X_T633, __global const float* restrict  X_T606, __global const float* restrict  X_T629, __global const float* restrict  X_I_232, __global const float* restrict  X_I_231)
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
      float LX_T606 = X_T606[gout_idx];
      float LX_T629 = X_T629[gout_idx];
      float LX_I_232 = X_I_232[(i4_gid + i4_tid)];
      float LX_I_231 = X_I_231[(i4_gid + i4_tid)];
      float LX_T630 = (LX_T606 + LX_T629);
      float LX_T632 = (LX_T630 - LX_I_232);
      float LX_T633 = (LX_T632 * LX_I_231);
      X_T630[gout_idx] = LX_T630;
      X_T633[gout_idx] = LX_T633;
    }
  }
}
