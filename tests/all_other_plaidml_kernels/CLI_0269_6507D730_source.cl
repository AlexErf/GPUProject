#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 176 }
// Out stride: { 8624, 1232, 176, 1 }
// Elementwise input X_T2349 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise input X_T2361 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise input X_I_871 shape: fp32(176):(1):704 bytes
// Elementwise input X_I_870 shape: fp32(176):(1):704 bytes
// Elementwise op: [[pid(Concatenate)]] X_T2362 = add(X_T2349, X_T2361)
// Elementwise op: [[pid(Sub)]] X_T2363 = sub(X_T2362, X_I_871)
// Elementwise op: [[pid(Mul)]] X_T2364 = mul(X_T2363, X_I_870)
// Tile size: { 1, 7, 1, 64 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 25872
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 224
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_908(__global float* restrict  X_T2364, __global const float* restrict  X_T2349, __global const float* restrict  X_T2361, __global const float* restrict  X_I_871, __global const float* restrict  X_I_870)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || ((i4_gid != 128) || (i4_tid < 16)));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i2_cond = (i2_tid < 7);
      if (i2_cond)
      {
        int gout_idx = (((1232 * i2_tid) + (176 * i3_gid)) + (i4_gid + i4));
        float LX_T2349 = X_T2349[gout_idx];
        float LX_T2361 = X_T2361[gout_idx];
        float LX_I_871 = X_I_871[(i4_gid + i4)];
        float LX_I_870 = X_I_870[(i4_gid + i4)];
        float LX_T2362 = (LX_T2349 + LX_T2361);
        float LX_T2363 = (LX_T2362 - LX_I_871);
        float LX_T2364 = (LX_T2363 * LX_I_870);
        X_T2364[gout_idx] = LX_T2364;
      }
    }
  }
}
