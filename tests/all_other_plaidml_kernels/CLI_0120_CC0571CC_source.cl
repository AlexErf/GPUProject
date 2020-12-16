#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 42, 42, 84 }
// Out stride: { 148176, 3528, 84, 1 }
// Elementwise input X_T309 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise input X_T313 shape: fp32(84):(1):336 bytes
// Elementwise input X_I_122 shape: fp32(84):(1):336 bytes
// Elementwise input X_T251 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T314 = div(X_T309, X_T313)
// Elementwise op: [[pid(Add, Switch)]] X_T315 = add(X_T314, X_I_122)
// Elementwise op: [[pid(Add)]] X_T316 = add(X_T251, X_T315)
// Tile size: { 1, 2, 2, 84 }
// Contraction output var shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Computed true ops: 444528
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_104(__global float* restrict  X_T316, __global const float* restrict  X_T309, __global const float* restrict  X_T313, __global const float* restrict  X_I_122, __global const float* restrict  X_T251)
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
      float LX_T309 = X_T309[gout_idx];
      float LX_T313 = X_T313[i4];
      float LX_I_122 = X_I_122[i4];
      float LX_T251 = X_T251[gout_idx];
      float LX_T314 = (LX_T309 / LX_T313);
      float LX_T315 = (LX_T314 + LX_I_122);
      float LX_T316 = (LX_T251 + LX_T315);
      X_T316[gout_idx] = LX_T316;
    }
  }
}
