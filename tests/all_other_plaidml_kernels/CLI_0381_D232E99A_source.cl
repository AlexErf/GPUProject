#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 25 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 800 }
// Out stride: { 156800, 11200, 800, 1 }
// Elementwise input X_T1000 shape: fp32(1, 14, 14, 800):(156800, 11200, 800, 1):612.5 KiB
// Elementwise input X_T1004 shape: fp32(800):(1):3.125 KiB
// Elementwise input X_I_380 shape: fp32(800):(1):3.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1005 = div(X_T1000, X_T1004)
// Elementwise op: [[pid(Add, Switch)]] X_T1006 = add(X_T1005, X_I_380)
// Elementwise op: X_T1007 = cmp_lt(X_T1006, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1008 = cond(X_T1007, X_T2, X_T1006)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 800):(156800, 11200, 800, 1):612.5 KiB
// Computed true ops: 627200
// Computed work groups: 175
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 25, 1
__kernel void kernel_c108_sdk_335(__global float* restrict  X_T1008, __global const float* restrict  X_T1000, __global const float* restrict  X_T1004, __global const float* restrict  X_I_380)
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
      int gout_idx = (((11200 * (i2_gid + i2_tid)) + (800 * i3)) + (i4_gid + i4_tid));
      float LX_T1000 = X_T1000[gout_idx];
      float LX_T1004 = X_T1004[(i4_gid + i4_tid)];
      float LX_I_380 = X_I_380[(i4_gid + i4_tid)];
      float LX_T1005 = (LX_T1000 / LX_T1004);
      float LX_T1006 = (LX_T1005 + LX_I_380);
      int LX_T1007 = (LX_T1006 < 0.0f);
      float LX_T1008 = select((float)LX_T1006, (float)0.0f, (int)LX_T1007);
      X_T1008[gout_idx] = LX_T1008;
    }
  }
}
