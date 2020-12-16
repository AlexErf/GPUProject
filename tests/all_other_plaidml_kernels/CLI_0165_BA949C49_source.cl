#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 224 }
// Out stride: { 175616, 6272, 224, 1 }
// Elementwise input X_T283 shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Elementwise input X_T306 shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Elementwise input X_I_121 shape: fp32(224):(1):896 bytes
// Elementwise input X_I_120 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(Concatenate)]] X_T307 = add(X_T283, X_T306)
// Elementwise op: [[pid(Sub)]] X_T309 = sub(X_T307, X_I_121)
// Elementwise op: [[pid(Mul)]] X_T310 = mul(X_T309, X_I_120)
// Tile size: { 1, 4, 1, 224 }
// Contraction output var shape: fp32(1, 28, 28, 224):(175616, 6272, 224, 1):686 KiB
// Computed true ops: 526848
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c68_sdk_92(__global float* restrict  X_T307, __global float* restrict  X_T310, __global const float* restrict  X_T283, __global const float* restrict  X_T306, __global const float* restrict  X_I_121, __global const float* restrict  X_I_120)
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
      float LX_T283 = X_T283[gout_idx];
      float LX_T306 = X_T306[gout_idx];
      float LX_I_121 = X_I_121[i4];
      float LX_I_120 = X_I_120[i4];
      float LX_T307 = (LX_T283 + LX_T306);
      float LX_T309 = (LX_T307 - LX_I_121);
      float LX_T310 = (LX_T309 * LX_I_120);
      X_T307[gout_idx] = LX_T307;
      X_T310[gout_idx] = LX_T310;
    }
  }
}
