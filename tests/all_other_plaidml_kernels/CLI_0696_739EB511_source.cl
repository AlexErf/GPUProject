#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1536 }
// Out stride: { 75264, 10752, 1536, 1 }
// Elementwise input X_T2276 shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Elementwise input X_T2299 shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Elementwise input X_I_893 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_892 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(Concatenate)]] X_T2300 = add(X_T2276, X_T2299)
// Elementwise op: [[pid(Sub)]] X_T2302 = sub(X_T2300, X_I_893)
// Elementwise op: [[pid(Mul)]] X_T2303 = mul(X_T2302, X_I_892)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1536):(75264, 10752, 1536, 1):294 KiB
// Computed true ops: 225792
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c124_sdk_797(__global float* restrict  X_T2300, __global float* restrict  X_T2303, __global const float* restrict  X_T2276, __global const float* restrict  X_T2299, __global const float* restrict  X_I_893, __global const float* restrict  X_I_892)
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
      int gout_idx = (((10752 * i2_gid) + (1536 * i3_tid)) + (i4_gid + i4));
      float LX_T2276 = X_T2276[gout_idx];
      float LX_T2299 = X_T2299[gout_idx];
      float LX_I_893 = X_I_893[(i4_gid + i4)];
      float LX_I_892 = X_I_892[(i4_gid + i4)];
      float LX_T2300 = (LX_T2276 + LX_T2299);
      float LX_T2302 = (LX_T2300 - LX_I_893);
      float LX_T2303 = (LX_T2302 * LX_I_892);
      X_T2300[gout_idx] = LX_T2300;
      X_T2303[gout_idx] = LX_T2303;
    }
  }
}
