#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 704 }
// Out stride: { 34496, 4928, 704, 1 }
// Elementwise input X_T1325 shape: fp32(1, 7, 7, 704):(34496, 4928, 704, 1):134.75 KiB
// Elementwise input X_T1329 shape: fp32(704):(1):2.75 KiB
// Elementwise input X_I_511 shape: fp32(704):(1):2.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1330 = div(X_T1325, X_T1329)
// Elementwise op: [[pid(Add, Switch)]] X_T1331 = add(X_T1330, X_I_511)
// Elementwise op: X_T1332 = cmp_lt(X_T1331, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1333 = cond(X_T1332, X_T2, X_T1331)
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
__kernel void kernel_c68_sdk_458(__global float* restrict  X_T1333, __global const float* restrict  X_T1325, __global const float* restrict  X_T1329, __global const float* restrict  X_I_511)
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
        float LX_T1325 = X_T1325[gout_idx];
        float LX_T1329 = X_T1329[(i4_gid + i4)];
        float LX_I_511 = X_I_511[(i4_gid + i4)];
        float LX_T1330 = (LX_T1325 / LX_T1329);
        float LX_T1331 = (LX_T1330 + LX_I_511);
        int LX_T1332 = (LX_T1331 < 0.0f);
        float LX_T1333 = select((float)LX_T1331, (float)0.0f, (int)LX_T1332);
        X_T1333[gout_idx] = LX_T1333;
      }
    }
  }
}
