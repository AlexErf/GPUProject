#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 512 }
// Out stride: { 100352, 7168, 512, 1 }
// Elementwise input X_T256 shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Elementwise input X_I_57 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_56 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Sub)]] X_T257 = sub(X_T256, X_I_57)
// Elementwise op: [[pid(Mul)]] X_T258 = mul(X_T257, X_I_56)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
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
__kernel void kernel_c25_sdk_64(__global float* restrict  X_T258, __global const float* restrict  X_T256, __global const float* restrict  X_I_57, __global const float* restrict  X_I_56)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((7168 * (i2_gid + i2_tid)) + (512 * i3)) + (i4_gid + i4_tid));
      float LX_T256 = X_T256[gout_idx];
      float LX_I_57 = X_I_57[(i4_gid + i4_tid)];
      float LX_I_56 = X_I_56[(i4_gid + i4_tid)];
      float LX_T257 = (LX_T256 - LX_I_57);
      float LX_T258 = (LX_T257 * LX_I_56);
      X_T258[gout_idx] = LX_T258;
    }
  }
}
