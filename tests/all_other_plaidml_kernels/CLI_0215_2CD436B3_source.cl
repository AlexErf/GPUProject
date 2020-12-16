#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 160 }
// Out stride: { 125440, 4480, 160, 1 }
// Elementwise input X_T249 shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Elementwise input X_T276 shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Elementwise input X_I_101 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_100 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(Concatenate)]] X_T277 = add(X_T249, X_T276)
// Elementwise op: [[pid(Sub)]] X_T279 = sub(X_T277, X_I_101)
// Elementwise op: [[pid(Mul)]] X_T280 = mul(X_T279, X_I_100)
// Tile size: { 1, 4, 1, 160 }
// Contraction output var shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Computed true ops: 376320
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 320
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 28, 1
__kernel void kernel_c108_sdk_74(__global float* restrict  X_T277, __global float* restrict  X_T280, __global const float* restrict  X_T249, __global const float* restrict  X_T276, __global const float* restrict  X_I_101, __global const float* restrict  X_I_100)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((4480 * (i2_gid + i2_tid)) + (160 * i3_gid)) + i4);
      float LX_T249 = X_T249[gout_idx];
      float LX_T276 = X_T276[gout_idx];
      float LX_I_101 = X_I_101[i4];
      float LX_I_100 = X_I_100[i4];
      float LX_T277 = (LX_T249 + LX_T276);
      float LX_T279 = (LX_T277 - LX_I_101);
      float LX_T280 = (LX_T279 * LX_I_100);
      X_T277[gout_idx] = LX_T277;
      X_T280[gout_idx] = LX_T280;
    }
  }
}
