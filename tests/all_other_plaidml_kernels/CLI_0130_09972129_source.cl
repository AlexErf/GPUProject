#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 22 }
// Out stride: { 17248, 616, 22, 1 }
// Elementwise input X_T386 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise input X_T390 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_140 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T391 = div(X_T386, X_T390)
// Elementwise op: [[pid(Add, Switch)]] X_T392 = add(X_T391, X_I_140)
// Tile size: { 1, 28, 1, 22 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 34496
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
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_132(__global float* restrict  X_T392, __global const float* restrict  X_T386, __global const float* restrict  X_T390, __global const float* restrict  X_I_140)
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
        float LX_T386 = X_T386[gout_idx];
        float LX_T390 = X_T390[i4_tid];
        float LX_I_140 = X_I_140[i4_tid];
        float LX_T391 = (LX_T386 / LX_T390);
        float LX_T392 = (LX_T391 + LX_I_140);
        X_T392[gout_idx] = LX_T392;
      }
    }
  }
}
