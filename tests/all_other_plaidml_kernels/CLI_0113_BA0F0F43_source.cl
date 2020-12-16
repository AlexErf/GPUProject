#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 22 }
// Out stride: { 68992, 1232, 22, 1 }
// Elementwise input X_T264 shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Elementwise input X_T276 shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Elementwise input X_I_122 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_121 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(Concatenate)]] X_T277 = add(X_T264, X_T276)
// Elementwise op: [[pid(Sub)]] X_T278 = sub(X_T277, X_I_122)
// Elementwise op: [[pid(Mul)]] X_T279 = mul(X_T278, X_I_121)
// Tile size: { 1, 56, 2, 22 }
// Contraction output var shape: fp32(1, 56, 56, 22):(68992, 1232, 22, 1):269.5 KiB
// Computed true ops: 206976
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_93(__global float* restrict  X_T279, __global const float* restrict  X_T264, __global const float* restrict  X_T276, __global const float* restrict  X_I_122, __global const float* restrict  X_I_121)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  int i4_cond = (i4_tid < 22);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
    {
      int i2 = ((4 * i2_lid) + i2_tid);
      int gout_idx = (((1232 * i2) + (22 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T264 = X_T264[gout_idx];
      float LX_T276 = X_T276[gout_idx];
      float LX_I_122 = X_I_122[i4_tid];
      float LX_I_121 = X_I_121[i4_tid];
      float LX_T277 = (LX_T264 + LX_T276);
      float LX_T278 = (LX_T277 - LX_I_122);
      float LX_T279 = (LX_T278 * LX_I_121);
      X_T279[gout_idx] = LX_T279;
    }
  }
}
