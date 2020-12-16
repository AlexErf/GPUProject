#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1024 }
// Out stride: { 50176, 7168, 1024, 1 }
// Elementwise input X_T420 shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Elementwise input X_I_9 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_8 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(Sub)]] X_T421 = sub(X_T420, X_I_9)
// Elementwise op: [[pid(Mul)]] X_T422 = mul(X_T421, X_I_8)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Computed true ops: 100352
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c25_sdk_107(__global float* restrict  X_T422, __global const float* restrict  X_T420, __global const float* restrict  X_I_9, __global const float* restrict  X_I_8)
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
      int gout_idx = (((7168 * i2_gid) + (1024 * i3_tid)) + (i4_gid + i4));
      float LX_T420 = X_T420[gout_idx];
      float LX_I_9 = X_I_9[(i4_gid + i4)];
      float LX_I_8 = X_I_8[(i4_gid + i4)];
      float LX_T421 = (LX_T420 - LX_I_9);
      float LX_T422 = (LX_T421 * LX_I_8);
      X_T422[gout_idx] = LX_T422;
    }
  }
}
