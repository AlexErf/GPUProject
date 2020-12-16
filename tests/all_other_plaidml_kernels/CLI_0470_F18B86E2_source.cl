#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 40 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1280 }
// Out stride: { 250880, 17920, 1280, 1 }
// Elementwise input X_T1374 shape: fp32(1, 14, 14, 1280):(250880, 17920, 1280, 1):980 KiB
// Elementwise input X_T1378 shape: fp32(1280):(1):5 KiB
// Elementwise input X_I_6 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1379 = div(X_T1374, X_T1378)
// Elementwise op: [[pid(Add, Switch)]] X_T1380 = add(X_T1379, X_I_6)
// Elementwise op: X_T1381 = cmp_lt(X_T1380, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1382 = cond(X_T1381, X_T2, X_T1380)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1280):(250880, 17920, 1280, 1):980 KiB
// Computed true ops: 1003520
// Computed work groups: 280
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 40, 1
__kernel void kernel_c108_sdk_469(__global float* restrict  X_T1382, __global const float* restrict  X_T1374, __global const float* restrict  X_T1378, __global const float* restrict  X_I_6)
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
      int gout_idx = (((17920 * (i2_gid + i2_tid)) + (1280 * i3)) + (i4_gid + i4_tid));
      float LX_T1374 = X_T1374[gout_idx];
      float LX_T1378 = X_T1378[(i4_gid + i4_tid)];
      float LX_I_6 = X_I_6[(i4_gid + i4_tid)];
      float LX_T1379 = (LX_T1374 / LX_T1378);
      float LX_T1380 = (LX_T1379 + LX_I_6);
      int LX_T1381 = (LX_T1380 < 0.0f);
      float LX_T1382 = select((float)LX_T1380, (float)0.0f, (int)LX_T1381);
      X_T1382[gout_idx] = LX_T1382;
    }
  }
}
