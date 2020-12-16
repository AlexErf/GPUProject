#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1824 }
// Out stride: { 89376, 12768, 1824, 1 }
// Elementwise input X_T2528 shape: fp32(1, 7, 7, 1824):(89376, 12768, 1824, 1):349.125 KiB
// Elementwise input X_T2532 shape: fp32(1824):(1):7.125 KiB
// Elementwise input X_I_981 shape: fp32(1824):(1):7.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2533 = div(X_T2528, X_T2532)
// Elementwise op: [[pid(Add, Switch)]] X_T2534 = add(X_T2533, X_I_981)
// Elementwise op: X_T2535 = cmp_lt(X_T2534, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2536 = cond(X_T2535, X_T2, X_T2534)
// Tile size: { 1, 1, 1, 1824 }
// Contraction output var shape: fp32(1, 7, 7, 1824):(89376, 12768, 1824, 1):349.125 KiB
// Computed true ops: 357504
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 684
// Computed mem write: 7296
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_881(__global float* restrict  X_T2536, __global const float* restrict  X_T2528, __global const float* restrict  X_T2532, __global const float* restrict  X_I_981)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 8; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 7) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((12768 * i2_gid) + (1824 * i3_gid)) + i4);
      float LX_T2528 = X_T2528[gout_idx];
      float LX_T2532 = X_T2532[i4];
      float LX_I_981 = X_I_981[i4];
      float LX_T2533 = (LX_T2528 / LX_T2532);
      float LX_T2534 = (LX_T2533 + LX_I_981);
      int LX_T2535 = (LX_T2534 < 0.0f);
      float LX_T2536 = select((float)LX_T2534, (float)0.0f, (int)LX_T2535);
      X_T2536[gout_idx] = LX_T2536;
    }
  }
}
