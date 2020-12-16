#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 19 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 608 }
// Out stride: { 119168, 8512, 608, 1 }
// Elementwise input X_T831 shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Elementwise input X_T854 shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Elementwise input X_I_322 shape: fp32(608):(1):2.375 KiB
// Elementwise input X_I_321 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T855 = add(X_T831, X_T854)
// Elementwise op: [[pid(Sub)]] X_T857 = sub(X_T855, X_I_322)
// Elementwise op: [[pid(Mul)]] X_T858 = mul(X_T857, X_I_321)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Computed true ops: 357504
// Computed work groups: 133
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 19, 1
__kernel void kernel_c124_sdk_278(__global float* restrict  X_T855, __global float* restrict  X_T858, __global const float* restrict  X_T831, __global const float* restrict  X_T854, __global const float* restrict  X_I_322, __global const float* restrict  X_I_321)
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
      int gout_idx = (((8512 * (i2_gid + i2_tid)) + (608 * i3)) + (i4_gid + i4_tid));
      float LX_T831 = X_T831[gout_idx];
      float LX_T854 = X_T854[gout_idx];
      float LX_I_322 = X_I_322[(i4_gid + i4_tid)];
      float LX_I_321 = X_I_321[(i4_gid + i4_tid)];
      float LX_T855 = (LX_T831 + LX_T854);
      float LX_T857 = (LX_T855 - LX_I_322);
      float LX_T858 = (LX_T857 * LX_I_321);
      X_T855[gout_idx] = LX_T855;
      X_T858[gout_idx] = LX_T858;
    }
  }
}
