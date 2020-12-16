#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 384 }
// Out stride: { 75264, 5376, 384, 1 }
// Elementwise input X_T655 shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Elementwise input X_T659 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_250 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T660 = div(X_T655, X_T659)
// Elementwise op: [[pid(Add, Switch)]] X_T661 = add(X_T660, X_I_250)
// Elementwise op: X_T662 = cmp_lt(X_T661, X_T2)
// Elementwise op: [[pid(Relu)]] X_T663 = cond(X_T662, X_T2, X_T661)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Computed true ops: 301056
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c68_sdk_218(__global float* restrict  X_T663, __global const float* restrict  X_T655, __global const float* restrict  X_T659, __global const float* restrict  X_I_250)
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
      int gout_idx = (((5376 * (i2_gid + i2_tid)) + (384 * i3)) + (i4_gid + i4_tid));
      float LX_T655 = X_T655[gout_idx];
      float LX_T659 = X_T659[(i4_gid + i4_tid)];
      float LX_I_250 = X_I_250[(i4_gid + i4_tid)];
      float LX_T660 = (LX_T655 / LX_T659);
      float LX_T661 = (LX_T660 + LX_I_250);
      int LX_T662 = (LX_T661 < 0.0f);
      float LX_T663 = select((float)LX_T661, (float)0.0f, (int)LX_T662);
      X_T663[gout_idx] = LX_T663;
    }
  }
}
