#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 768 }
// Out stride: { 37632, 5376, 768, 1 }
// Elementwise input X_T1375 shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Elementwise input X_T1379 shape: fp32(768):(1):3 KiB
// Elementwise input X_I_531 shape: fp32(768):(1):3 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1380 = div(X_T1375, X_T1379)
// Elementwise op: [[pid(Add, Switch)]] X_T1381 = add(X_T1380, X_I_531)
// Elementwise op: X_T1382 = cmp_lt(X_T1381, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1383 = cond(X_T1382, X_T2, X_T1381)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 768):(37632, 5376, 768, 1):147 KiB
// Computed true ops: 150528
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c68_sdk_476(__global float* restrict  X_T1383, __global const float* restrict  X_T1375, __global const float* restrict  X_T1379, __global const float* restrict  X_I_531)
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
      float LX_T1375 = X_T1375[gout_idx];
      float LX_T1379 = X_T1379[(i4_gid + i4)];
      float LX_I_531 = X_I_531[(i4_gid + i4)];
      float LX_T1380 = (LX_T1375 / LX_T1379);
      float LX_T1381 = (LX_T1380 + LX_I_531);
      int LX_T1382 = (LX_T1381 < 0.0f);
      float LX_T1383 = select((float)LX_T1381, (float)0.0f, (int)LX_T1382);
      X_T1383[gout_idx] = LX_T1383;
    }
  }
}
