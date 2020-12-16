#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 768 }
// Out stride: { 37632, 5376, 768, 1 }
// Elementwise input X_T1348 shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Elementwise input X_T1371 shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Elementwise input X_I_533 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_532 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1372 = add(X_T1348, X_T1371)
// Elementwise op: [[pid(Sub)]] X_T1374 = sub(X_T1372, X_I_533)
// Elementwise op: [[pid(Mul)]] X_T1375 = mul(X_T1374, X_I_532)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Computed true ops: 112896
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c68_sdk_473(__global float* restrict  X_T1372, __global float* restrict  X_T1375, __global const float* restrict  X_T1348, __global const float* restrict  X_T1371, __global const float* restrict  X_I_533, __global const float* restrict  X_I_532)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((5376 * i2_gid) + (768 * i3_tid)) + (i4_gid + i4));
      float LX_T1348 = X_T1348[gout_idx];
      float LX_T1371 = X_T1371[gout_idx];
      float LX_I_533 = X_I_533[(i4_gid + i4)];
      float LX_I_532 = X_I_532[(i4_gid + i4)];
      float LX_T1372 = (LX_T1348 + LX_T1371);
      float LX_T1374 = (LX_T1372 - LX_I_533);
      float LX_T1375 = (LX_T1374 * LX_I_532);
      X_T1372[gout_idx] = LX_T1372;
      X_T1375[gout_idx] = LX_T1375;
    }
  }
}
