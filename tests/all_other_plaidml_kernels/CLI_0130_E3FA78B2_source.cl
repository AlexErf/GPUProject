#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 84 }
// Out stride: { 148176, 3528, 84, 1 }
// Elementwise input X_T387 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise input X_T391 shape: fp32(84):(1):336 bytes
// Elementwise input X_I_152 shape: fp32(84):(1):336 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T392 = div(X_T387, X_T391)
// Elementwise op: [[pid(Add, Switch)]] X_T393 = add(X_T392, X_I_152)
// Tile size: { 1, 2, 2, 84 }
// Contraction output var shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Computed true ops: 296352
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_132(__global float* restrict  X_T393, __global const float* restrict  X_T387, __global const float* restrict  X_T391, __global const float* restrict  X_I_152)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 20));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((3528 * (i2_gid + i2_tid)) + (84 * (i3_gid + i3_tid))) + i4);
      float LX_T387 = X_T387[gout_idx];
      float LX_T391 = X_T391[i4];
      float LX_I_152 = X_I_152[i4];
      float LX_T392 = (LX_T387 / LX_T391);
      float LX_T393 = (LX_T392 + LX_I_152);
      X_T393[gout_idx] = LX_T393;
    }
  }
}
