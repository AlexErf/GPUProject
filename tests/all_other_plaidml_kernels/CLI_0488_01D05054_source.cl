#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 704 }
// Out stride: { 34496, 4928, 704, 1 }
// Elementwise input X_T1445 shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Elementwise input X_T1449 shape: fp32(704):(1):2.75 KiB
// Elementwise input X_I_551 shape: fp32(704):(1):2.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1450 = div(X_T1445, X_T1449)
// Elementwise op: [[pid(Add, Switch)]] X_T1451 = add(X_T1450, X_I_551)
// Elementwise op: X_T1452 = cmp_lt(X_T1451, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1453 = cond(X_T1452, X_T2, X_T1451)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Computed true ops: 137984
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
__kernel void kernel_c108_sdk_494(__global float* restrict  X_T1453, __global const float* restrict  X_T1445, __global const float* restrict  X_T1449, __global const float* restrict  X_I_551)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 640));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((4928 * i2_gid) + (704 * i3_tid)) + (i4_gid + i4));
        float LX_T1445 = X_T1445[gout_idx];
        float LX_T1449 = X_T1449[(i4_gid + i4)];
        float LX_I_551 = X_I_551[(i4_gid + i4)];
        float LX_T1450 = (LX_T1445 / LX_T1449);
        float LX_T1451 = (LX_T1450 + LX_I_551);
        int LX_T1452 = (LX_T1451 < 0.0f);
        float LX_T1453 = select((float)LX_T1451, (float)0.0f, (int)LX_T1452);
        X_T1453[gout_idx] = LX_T1453;
      }
    }
  }
}
