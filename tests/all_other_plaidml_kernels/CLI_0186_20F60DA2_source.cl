#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 21, 21, 336 }
// Out stride: { 148176, 7056, 336, 1 }
// Elementwise input X_T1597 shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Elementwise input X_T1601 shape: fp32(336):(1):1.3125 KiB
// Elementwise input X_I_598 shape: fp32(336):(1):1.3125 KiB
// Elementwise input X_T1559 shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1602 = div(X_T1597, X_T1601)
// Elementwise op: [[pid(Add, Switch)]] X_T1603 = add(X_T1602, X_I_598)
// Elementwise op: [[pid(Add)]] X_T1604 = add(X_T1559, X_T1603)
// Tile size: { 1, 2, 2, 128 }
// Contraction output var shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Computed true ops: 444528
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 256
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_608(__global float* restrict  X_T1604, __global const float* restrict  X_T1597, __global const float* restrict  X_T1601, __global const float* restrict  X_I_598, __global const float* restrict  X_T1559)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = (get_group_id(2) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || ((i4_gid != 256) || (i4_tid < 16)));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int i3_cond = ((i3_gid != 20) || (i3_tid < 1));
      if (i3_cond)
      {
        int i2_cond = ((i2_gid != 20) || (i2_tid < 1));
        if (i2_cond)
        {
          int gout_idx = (((7056 * (i2_gid + i2_tid)) + (336 * (i3_gid + i3_tid))) + (i4_gid + i4));
          float LX_T1597 = X_T1597[gout_idx];
          float LX_T1601 = X_T1601[(i4_gid + i4)];
          float LX_I_598 = X_I_598[(i4_gid + i4)];
          float LX_T1559 = X_T1559[gout_idx];
          float LX_T1602 = (LX_T1597 / LX_T1601);
          float LX_T1603 = (LX_T1602 + LX_I_598);
          float LX_T1604 = (LX_T1559 + LX_T1603);
          X_T1604[gout_idx] = LX_T1604;
        }
      }
    }
  }
}
