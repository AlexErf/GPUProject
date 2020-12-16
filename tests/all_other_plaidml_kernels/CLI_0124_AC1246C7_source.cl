#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 96 }
// Out stride: { 18816, 1344, 96, 1 }
// Elementwise input X_T472 shape: fp32(1, 14, 14, 96):(18816, 1344, 96, 1):73.5 KiB
// Elementwise input X_T476 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_30 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T477 = div(X_T472, X_T476)
// Elementwise op: [[pid(Add, Switch)]] X_T478 = add(X_T477, X_I_30)
// Tile size: { 1, 2, 2, 96 }
// Contraction output var shape: fp32(1, 14, 14, 96):(18816, 1344, 96, 1):73.5 KiB
// Computed true ops: 37632
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_126(__global float* restrict  X_T478, __global const float* restrict  X_T472, __global const float* restrict  X_T476, __global const float* restrict  X_I_30)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((1344 * (i2_gid + i2_tid)) + (96 * (i3_gid + i3_tid))) + i4);
      float LX_T472 = X_T472[gout_idx];
      float LX_T476 = X_T476[i4];
      float LX_I_30 = X_I_30[i4];
      float LX_T477 = (LX_T472 / LX_T476);
      float LX_T478 = (LX_T477 + LX_I_30);
      X_T478[gout_idx] = LX_T478;
    }
  }
}
