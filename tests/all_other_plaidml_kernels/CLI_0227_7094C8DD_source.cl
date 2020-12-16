#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 128 }
// Out stride: { 25088, 1792, 128, 1 }
// Elementwise input X_T566 shape: fp32(1, 14, 14, 128):(25088, 1792, 128, 1):98 KiB
// Elementwise input X_T570 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_206 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T571 = div(X_T566, X_T570)
// Elementwise op: [[pid(Add, Switch)]] X_T572 = add(X_T571, X_I_206)
// Elementwise op: X_T573 = cmp_lt(X_T572, X_T2)
// Elementwise op: [[pid(Relu)]] X_T574 = cond(X_T573, X_T2, X_T572)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 128):(25088, 1792, 128, 1):98 KiB
// Computed true ops: 100352
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c68_sdk_185(__global float* restrict  X_T574, __global const float* restrict  X_T566, __global const float* restrict  X_T570, __global const float* restrict  X_I_206)
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
      int gout_idx = (((1792 * (i2_gid + i2_tid)) + (128 * i3)) + (i4_gid + i4_tid));
      float LX_T566 = X_T566[gout_idx];
      float LX_T570 = X_T570[(i4_gid + i4_tid)];
      float LX_I_206 = X_I_206[(i4_gid + i4_tid)];
      float LX_T571 = (LX_T566 / LX_T570);
      float LX_T572 = (LX_T571 + LX_I_206);
      int LX_T573 = (LX_T572 < 0.0f);
      float LX_T574 = select((float)LX_T572, (float)0.0f, (int)LX_T573);
      X_T574[gout_idx] = LX_T574;
    }
  }
}
