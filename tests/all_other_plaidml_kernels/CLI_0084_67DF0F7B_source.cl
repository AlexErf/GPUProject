#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 96 }
// Out stride: { 301056, 5376, 96, 1 }
// Elementwise input X_T109 shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Elementwise input X_I_72 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_71 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(Sub)]] X_T110 = sub(X_T109, X_I_72)
// Elementwise op: [[pid(Mul)]] X_T111 = mul(X_T110, X_I_71)
// Tile size: { 1, 8, 1, 96 }
// Contraction output var shape: fp32(1, 56, 56, 96):(301056, 5376, 96, 1):1176 KiB
// Computed true ops: 602112
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 288
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 56, 1
__kernel void kernel_c43_sdk_23(__global float* restrict  X_T111, __global const float* restrict  X_T109, __global const float* restrict  X_I_72, __global const float* restrict  X_I_71)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 8);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    int gout_idx = (((5376 * (i2_gid + i2_tid)) + (96 * i3_gid)) + i4);
    float LX_T109 = X_T109[gout_idx];
    float LX_I_72 = X_I_72[i4];
    float LX_I_71 = X_I_71[i4];
    float LX_T110 = (LX_T109 - LX_I_72);
    float LX_T111 = (LX_T110 * LX_I_71);
    X_T111[gout_idx] = LX_T111;
  }
}
