#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 192 }
// Out stride: { 37632, 2688, 192, 1 }
// Elementwise input X_T307 shape: fp32(1, 14, 14, 192):(37632, 2688, 192, 1):147 KiB
// Elementwise input X_I_48 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_47 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Sub)]] X_T308 = sub(X_T307, X_I_48)
// Elementwise op: [[pid(Mul)]] X_T309 = mul(X_T308, X_I_47)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 192):(37632, 2688, 192, 1):147 KiB
// Computed true ops: 75264
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c43_sdk_78(__global float* restrict  X_T309, __global const float* restrict  X_T307, __global const float* restrict  X_I_48, __global const float* restrict  X_I_47)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((2688 * (i2_gid + i2_tid)) + (192 * i3)) + (i4_gid + i4_tid));
      float LX_T307 = X_T307[gout_idx];
      float LX_I_48 = X_I_48[(i4_gid + i4_tid)];
      float LX_I_47 = X_I_47[(i4_gid + i4_tid)];
      float LX_T308 = (LX_T307 - LX_I_48);
      float LX_T309 = (LX_T308 * LX_I_47);
      X_T309[gout_idx] = LX_T309;
    }
  }
}
