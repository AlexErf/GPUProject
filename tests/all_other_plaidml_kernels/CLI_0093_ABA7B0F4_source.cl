#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 144 }
// Out stride: { 451584, 8064, 144, 1 }
// Elementwise input X_T150 shape: fp32(1, 56, 56, 144):(451584, 8064, 144, 1):1764 KiB
// Elementwise input X_I_103 shape: fp32(144):(1):576 bytes
// Elementwise input X_I_102 shape: fp32(144):(1):576 bytes
// Elementwise op: [[pid(Sub)]] X_T151 = sub(X_T150, X_I_103)
// Elementwise op: [[pid(Mul)]] X_T152 = mul(X_T151, X_I_102)
// Tile size: { 1, 4, 4, 144 }
// Contraction output var shape: fp32(1, 56, 56, 144):(451584, 8064, 144, 1):1764 KiB
// Computed true ops: 903168
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 960
// Computed mem write: 10240
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c43_sdk_34(__global float* restrict  X_T152, __global const float* restrict  X_T150, __global const float* restrict  X_I_103, __global const float* restrict  X_I_102)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 4) || (i4_tid < 16));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
      {
        int i2 = ((2 * i2_lid) + i2_tid);
        int gout_idx = (((8064 * (i2_gid + i2)) + (144 * (i3_gid + i3_tid))) + i4);
        float LX_T150 = X_T150[gout_idx];
        float LX_I_103 = X_I_103[i4];
        float LX_I_102 = X_I_102[i4];
        float LX_T151 = (LX_T150 - LX_I_103);
        float LX_T152 = (LX_T151 * LX_I_102);
        X_T152[gout_idx] = LX_T152;
      }
    }
  }
}
