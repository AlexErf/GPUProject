#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 736 }
// Out stride: { 36064, 5152, 736, 1 }
// Elementwise input X_T1350 shape: fp32(1, 7, 7, 736):(36064, 5152, 736, 1):140.875 KiB
// Elementwise input X_T1354 shape: fp32(736):(1):2.875 KiB
// Elementwise input X_I_521 shape: fp32(736):(1):2.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1355 = div(X_T1350, X_T1354)
// Elementwise op: [[pid(Add, Switch)]] X_T1356 = add(X_T1355, X_I_521)
// Elementwise op: X_T1357 = cmp_lt(X_T1356, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1358 = cond(X_T1357, X_T2, X_T1356)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 736):(36064, 5152, 736, 1):140.875 KiB
// Computed true ops: 144256
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
__kernel void kernel_c68_sdk_467(__global float* restrict  X_T1358, __global const float* restrict  X_T1350, __global const float* restrict  X_T1354, __global const float* restrict  X_I_521)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 640));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((5152 * i2_gid) + (736 * i3_tid)) + (i4_gid + i4));
        float LX_T1350 = X_T1350[gout_idx];
        float LX_T1354 = X_T1354[(i4_gid + i4)];
        float LX_I_521 = X_I_521[(i4_gid + i4)];
        float LX_T1355 = (LX_T1350 / LX_T1354);
        float LX_T1356 = (LX_T1355 + LX_I_521);
        int LX_T1357 = (LX_T1356 < 0.0f);
        float LX_T1358 = select((float)LX_T1356, (float)0.0f, (int)LX_T1357);
        X_T1358[gout_idx] = LX_T1358;
      }
    }
  }
}
