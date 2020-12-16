#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 22 }
// Out stride: { 17248, 616, 22, 1 }
// Elementwise input X_T448 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise input X_T452 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_164 shape: fp32(22):(1):88 bytes
// Elementwise input X_T249 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T453 = div(X_T448, X_T452)
// Elementwise op: [[pid(Add, Switch)]] X_T454 = add(X_T453, X_I_164)
// Elementwise op: [[pid(Add)]] X_T455 = add(X_T454, X_T249)
// Tile size: { 1, 28, 1, 22 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 51744
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
__kernel void kernel_c42_sdk_157(__global float* restrict  X_T455, __global const float* restrict  X_T448, __global const float* restrict  X_T452, __global const float* restrict  X_I_164, __global const float* restrict  X_T249)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 22);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
    {
      int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
      if (i2_cond)
      {
        int i2 = ((8 * i2_lid) + i2_tid);
        int gout_idx = (((616 * i2) + (22 * i3_gid)) + i4_tid);
        float LX_T448 = X_T448[gout_idx];
        float LX_T452 = X_T452[i4_tid];
        float LX_I_164 = X_I_164[i4_tid];
        float LX_T249 = X_T249[gout_idx];
        float LX_T453 = (LX_T448 / LX_T452);
        float LX_T454 = (LX_T453 + LX_I_164);
        float LX_T455 = (LX_T454 + LX_T249);
        X_T455[gout_idx] = LX_T455;
      }
    }
  }
}
