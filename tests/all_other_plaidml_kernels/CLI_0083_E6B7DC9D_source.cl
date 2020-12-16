#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 512 }
// Out stride: { 25088, 3584, 512, 1 }
// Elementwise input X_T393 shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Elementwise input X_I_17 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_16 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Sub)]] X_T394 = sub(X_T393, X_I_17)
// Elementwise op: [[pid(Mul)]] X_T395 = mul(X_T394, X_I_16)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Computed true ops: 50176
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
// gwork = 1792, 4, 1
__kernel void kernel_c25_sdk_100(__global float* restrict  X_T395, __global const float* restrict  X_T393, __global const float* restrict  X_I_17, __global const float* restrict  X_I_16)
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
      int gout_idx = (((3584 * i2_gid) + (512 * i3_tid)) + (i4_gid + i4));
      float LX_T393 = X_T393[gout_idx];
      float LX_I_17 = X_I_17[(i4_gid + i4)];
      float LX_I_16 = X_I_16[(i4_gid + i4)];
      float LX_T394 = (LX_T393 - LX_I_17);
      float LX_T395 = (LX_T394 * LX_I_16);
      X_T395[gout_idx] = LX_T395;
    }
  }
}
