#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1056 }
// Out stride: { 51744, 7392, 1056, 1 }
// Elementwise input X_T1720 shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Elementwise input X_T1724 shape: fp32(1056):(1):4.125 KiB
// Elementwise input X_I_661 shape: fp32(1056):(1):4.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1725 = div(X_T1720, X_T1724)
// Elementwise op: [[pid(Add, Switch)]] X_T1726 = add(X_T1725, X_I_661)
// Elementwise op: X_T1727 = cmp_lt(X_T1726, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1728 = cond(X_T1727, X_T2, X_T1726)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1056):(51744, 7392, 1056, 1):202.125 KiB
// Computed true ops: 206976
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c108_sdk_593(__global float* restrict  X_T1728, __global const float* restrict  X_T1720, __global const float* restrict  X_T1724, __global const float* restrict  X_I_661)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 1024));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((7392 * i2_gid) + (1056 * i3_tid)) + (i4_gid + i4));
        float LX_T1720 = X_T1720[gout_idx];
        float LX_T1724 = X_T1724[(i4_gid + i4)];
        float LX_I_661 = X_I_661[(i4_gid + i4)];
        float LX_T1725 = (LX_T1720 / LX_T1724);
        float LX_T1726 = (LX_T1725 + LX_I_661);
        int LX_T1727 = (LX_T1726 < 0.0f);
        float LX_T1728 = select((float)LX_T1726, (float)0.0f, (int)LX_T1727);
        X_T1728[gout_idx] = LX_T1728;
      }
    }
  }
}