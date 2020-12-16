#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 33 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1056 }
// Out stride: { 206976, 14784, 1056, 1 }
// Elementwise input X_T1208 shape: fp32(1, 14, 14, 1056):(206976, 14784, 1056, 1):808.5 KiB
// Elementwise input X_T1212 shape: fp32(1056):(1):4.125 KiB
// Elementwise input X_I_460 shape: fp32(1056):(1):4.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1213 = div(X_T1208, X_T1212)
// Elementwise op: [[pid(Add, Switch)]] X_T1214 = add(X_T1213, X_I_460)
// Elementwise op: X_T1215 = cmp_lt(X_T1214, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1216 = cond(X_T1215, X_T2, X_T1214)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1056):(206976, 14784, 1056, 1):808.5 KiB
// Computed true ops: 827904
// Computed work groups: 231
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 33, 1
__kernel void kernel_c124_sdk_407(__global float* restrict  X_T1216, __global const float* restrict  X_T1208, __global const float* restrict  X_T1212, __global const float* restrict  X_I_460)
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
      int gout_idx = (((14784 * (i2_gid + i2_tid)) + (1056 * i3)) + (i4_gid + i4_tid));
      float LX_T1208 = X_T1208[gout_idx];
      float LX_T1212 = X_T1212[(i4_gid + i4_tid)];
      float LX_I_460 = X_I_460[(i4_gid + i4_tid)];
      float LX_T1213 = (LX_T1208 / LX_T1212);
      float LX_T1214 = (LX_T1213 + LX_I_460);
      int LX_T1215 = (LX_T1214 < 0.0f);
      float LX_T1216 = select((float)LX_T1214, (float)0.0f, (int)LX_T1215);
      X_T1216[gout_idx] = LX_T1216;
    }
  }
}
