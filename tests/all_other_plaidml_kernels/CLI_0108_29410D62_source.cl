#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 32 }
// Out stride: { 25088, 896, 32, 1 }
// Elementwise input X_T245 shape: fp32(1, 28, 28, 32):(25088, 896, 32, 1):98 KiB
// Elementwise input X_T249 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_115 shape: fp32(32):(1):128 bytes
// Elementwise input X_T210 shape: fp32(1, 28, 28, 32):(25088, 896, 32, 1):98 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T250 = div(X_T245, X_T249)
// Elementwise op: [[pid(Add, Switch)]] X_T251 = add(X_T250, X_I_115)
// Elementwise op: [[pid(Add)]] X_T252 = add(X_T210, X_T251)
// Tile size: { 1, 28, 1, 32 }
// Contraction output var shape: fp32(1, 28, 28, 32):(25088, 896, 32, 1):98 KiB
// Computed true ops: 75264
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c43_sdk_61(__global float* restrict  X_T252, __global const float* restrict  X_T245, __global const float* restrict  X_T249, __global const float* restrict  X_I_115, __global const float* restrict  X_T210)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
  {
    int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
    if (i2_cond)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((896 * i2) + (32 * i3_gid)) + i4_tid);
      float LX_T245 = X_T245[gout_idx];
      float LX_T249 = X_T249[i4_tid];
      float LX_I_115 = X_I_115[i4_tid];
      float LX_T210 = X_T210[gout_idx];
      float LX_T250 = (LX_T245 / LX_T249);
      float LX_T251 = (LX_T250 + LX_I_115);
      float LX_T252 = (LX_T210 + LX_T251);
      X_T252[gout_idx] = LX_T252;
    }
  }
}
