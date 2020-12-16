#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1184 }
// Out stride: { 232064, 16576, 1184, 1 }
// Elementwise input X_T1281 shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Elementwise input X_T1304 shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Elementwise input X_I_502 shape: fp32(1184):(1):4.625 KiB
// Elementwise input X_I_501 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1305 = add(X_T1281, X_T1304)
// Elementwise op: [[pid(Sub)]] X_T1307 = sub(X_T1305, X_I_502)
// Elementwise op: [[pid(Mul)]] X_T1308 = mul(X_T1307, X_I_501)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1184):(232064, 16576, 1184, 1):906.5 KiB
// Computed true ops: 696192
// Computed work groups: 259
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 37, 1
__kernel void kernel_c124_sdk_440(__global float* restrict  X_T1305, __global float* restrict  X_T1308, __global const float* restrict  X_T1281, __global const float* restrict  X_T1304, __global const float* restrict  X_I_502, __global const float* restrict  X_I_501)
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
      int gout_idx = (((16576 * (i2_gid + i2_tid)) + (1184 * i3)) + (i4_gid + i4_tid));
      float LX_T1281 = X_T1281[gout_idx];
      float LX_T1304 = X_T1304[gout_idx];
      float LX_I_502 = X_I_502[(i4_gid + i4_tid)];
      float LX_I_501 = X_I_501[(i4_gid + i4_tid)];
      float LX_T1305 = (LX_T1281 + LX_T1304);
      float LX_T1307 = (LX_T1305 - LX_I_502);
      float LX_T1308 = (LX_T1307 * LX_I_501);
      X_T1305[gout_idx] = LX_T1305;
      X_T1308[gout_idx] = LX_T1308;
    }
  }
}
