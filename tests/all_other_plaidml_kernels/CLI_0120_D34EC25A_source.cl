#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 22 }
// Out stride: { 17248, 616, 22, 1 }
// Elementwise input X_T307 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise input X_T311 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_110 shape: fp32(22):(1):88 bytes
// Elementwise input X_T249 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T312 = div(X_T307, X_T311)
// Elementwise op: [[pid(Add, Switch)]] X_T313 = add(X_T312, X_I_110)
// Elementwise op: [[pid(Add)]] X_T314 = add(X_T249, X_T313)
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
__kernel void kernel_c42_sdk_104(__global float* restrict  X_T314, __global const float* restrict  X_T307, __global const float* restrict  X_T311, __global const float* restrict  X_I_110, __global const float* restrict  X_T249)
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
        float LX_T307 = X_T307[gout_idx];
        float LX_T311 = X_T311[i4_tid];
        float LX_I_110 = X_I_110[i4_tid];
        float LX_T249 = X_T249[gout_idx];
        float LX_T312 = (LX_T307 / LX_T311);
        float LX_T313 = (LX_T312 + LX_I_110);
        float LX_T314 = (LX_T249 + LX_T313);
        X_T314[gout_idx] = LX_T314;
      }
    }
  }
}
