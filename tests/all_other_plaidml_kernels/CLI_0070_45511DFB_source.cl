#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 2048 }
// Out stride: { 100352, 14336, 2048, 1 }
// Elementwise input X_T587 shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Elementwise input X_T591 shape: fp32(2048):(1):8 KiB
// Elementwise input X_I_32 shape: fp32(2048):(1):8 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T592 = div(X_T587, X_T591)
// Elementwise op: [[pid(Add, Switch)]] X_T593 = add(X_T592, X_I_32)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Computed true ops: 200704
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 16, 1
__kernel void kernel_c29_sdk_140(__global float* restrict  X_T593, __global const float* restrict  X_T587, __global const float* restrict  X_T591, __global const float* restrict  X_I_32)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((14336 * i2_gid) + (2048 * i3_tid)) + (i4_gid + i4));
      float LX_T587 = X_T587[gout_idx];
      float LX_T591 = X_T591[(i4_gid + i4)];
      float LX_I_32 = X_I_32[(i4_gid + i4)];
      float LX_T592 = (LX_T587 / LX_T591);
      float LX_T593 = (LX_T592 + LX_I_32);
      X_T593[gout_idx] = LX_T593;
    }
  }
}
