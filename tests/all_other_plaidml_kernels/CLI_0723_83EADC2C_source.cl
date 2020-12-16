#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1696 }
// Out stride: { 83104, 11872, 1696, 1 }
// Elementwise input X_T2428 shape: fp32(1, 7, 7, 1696):(83104, 11872, 1696, 1):324.625 KiB
// Elementwise input X_T2432 shape: fp32(1696):(1):6.625 KiB
// Elementwise input X_I_941 shape: fp32(1696):(1):6.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2433 = div(X_T2428, X_T2432)
// Elementwise op: [[pid(Add, Switch)]] X_T2434 = add(X_T2433, X_I_941)
// Elementwise op: X_T2435 = cmp_lt(X_T2434, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2436 = cond(X_T2435, X_T2, X_T2434)
// Tile size: { 1, 1, 1, 1696 }
// Contraction output var shape: fp32(1, 7, 7, 1696):(83104, 11872, 1696, 1):324.625 KiB
// Computed true ops: 332416
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 636
// Computed mem write: 6784
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_845(__global float* restrict  X_T2436, __global const float* restrict  X_T2428, __global const float* restrict  X_T2432, __global const float* restrict  X_I_941)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 160));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11872 * i2_gid) + (1696 * i3_gid)) + i4);
      float LX_T2428 = X_T2428[gout_idx];
      float LX_T2432 = X_T2432[i4];
      float LX_I_941 = X_I_941[i4];
      float LX_T2433 = (LX_T2428 / LX_T2432);
      float LX_T2434 = (LX_T2433 + LX_I_941);
      int LX_T2435 = (LX_T2434 < 0.0f);
      float LX_T2436 = select((float)LX_T2434, (float)0.0f, (int)LX_T2435);
      X_T2436[gout_idx] = LX_T2436;
    }
  }
}
