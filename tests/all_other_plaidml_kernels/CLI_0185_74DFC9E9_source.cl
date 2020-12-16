#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 160 }
// Out stride: { 501760, 8960, 160, 1 }
// Elementwise input X_T132 shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Elementwise input X_T155 shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Elementwise input X_I_60 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_59 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Concatenate)]] X_T156 = add(X_T132, X_T155)
// Elementwise op: [[pid(Sub)]] X_T158 = sub(X_T156, X_I_60)
// Elementwise op: [[pid(Mul)]] X_T159 = mul(X_T158, X_I_59)
// Tile size: { 1, 4, 4, 160 }
// Contraction output var shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Computed true ops: 1505280
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 1280
// Computed mem write: 20480
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c108_sdk_32(__global float* restrict  X_T156, __global float* restrict  X_T159, __global const float* restrict  X_T132, __global const float* restrict  X_T155, __global const float* restrict  X_I_60, __global const float* restrict  X_I_59)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((8960 * (i2_gid + i2)) + (160 * (i3_gid + i3_tid))) + i4);
      float LX_T132 = X_T132[gout_idx];
      float LX_T155 = X_T155[gout_idx];
      float LX_I_60 = X_I_60[i4];
      float LX_I_59 = X_I_59[i4];
      float LX_T156 = (LX_T132 + LX_T155);
      float LX_T158 = (LX_T156 - LX_I_60);
      float LX_T159 = (LX_T158 * LX_I_59);
      X_T156[gout_idx] = LX_T156;
      X_T159[gout_idx] = LX_T159;
    }
  }
}
