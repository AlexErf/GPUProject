#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 224 }
// Out stride: { 175616, 6272, 224, 1 }
// Elementwise input X_T310 shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Elementwise input X_T314 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_119 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T315 = div(X_T310, X_T314)
// Elementwise op: [[pid(Add, Switch)]] X_T316 = add(X_T315, X_I_119)
// Elementwise op: X_T317 = cmp_lt(X_T316, X_T2)
// Elementwise op: [[pid(Relu)]] X_T318 = cond(X_T317, X_T2, X_T316)
// Tile size: { 1, 4, 1, 224 }
// Contraction output var shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Computed true ops: 702464
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c68_sdk_95(__global float* restrict  X_T318, __global const float* restrict  X_T310, __global const float* restrict  X_T314, __global const float* restrict  X_I_119)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((6272 * (i2_gid + i2_tid)) + (224 * i3_gid)) + i4);
      float LX_T310 = X_T310[gout_idx];
      float LX_T314 = X_T314[i4];
      float LX_I_119 = X_I_119[i4];
      float LX_T315 = (LX_T310 / LX_T314);
      float LX_T316 = (LX_T315 + LX_I_119);
      int LX_T317 = (LX_T316 < 0.0f);
      float LX_T318 = select((float)LX_T316, (float)0.0f, (int)LX_T317);
      X_T318[gout_idx] = LX_T318;
    }
  }
}
