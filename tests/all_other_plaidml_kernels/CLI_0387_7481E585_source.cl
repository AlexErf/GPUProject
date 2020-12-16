#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 22 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 704 }
// Out stride: { 137984, 9856, 704, 1 }
// Elementwise input X_T933 shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Elementwise input X_T937 shape: fp32(704):(1):2.75 KiB
// Elementwise input X_I_350 shape: fp32(704):(1):2.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T938 = div(X_T933, X_T937)
// Elementwise op: [[pid(Add, Switch)]] X_T939 = add(X_T938, X_I_350)
// Elementwise op: X_T940 = cmp_lt(X_T939, X_T2)
// Elementwise op: [[pid(Relu)]] X_T941 = cond(X_T940, X_T2, X_T939)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Computed true ops: 551936
// Computed work groups: 154
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 22, 1
__kernel void kernel_c124_sdk_308(__global float* restrict  X_T941, __global const float* restrict  X_T933, __global const float* restrict  X_T937, __global const float* restrict  X_I_350)
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
      int gout_idx = (((9856 * (i2_gid + i2_tid)) + (704 * i3)) + (i4_gid + i4_tid));
      float LX_T933 = X_T933[gout_idx];
      float LX_T937 = X_T937[(i4_gid + i4_tid)];
      float LX_I_350 = X_I_350[(i4_gid + i4_tid)];
      float LX_T938 = (LX_T933 / LX_T937);
      float LX_T939 = (LX_T938 + LX_I_350);
      int LX_T940 = (LX_T939 < 0.0f);
      float LX_T941 = select((float)LX_T939, (float)0.0f, (int)LX_T940);
      X_T941[gout_idx] = LX_T941;
    }
  }
}
