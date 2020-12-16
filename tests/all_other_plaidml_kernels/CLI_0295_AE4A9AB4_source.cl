#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 480 }
// Out stride: { 376320, 13440, 480, 1 }
// Elementwise input X_T511 shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Elementwise input X_T534 shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Elementwise input X_I_201 shape: fp32(480):(1):1.875 KiB
// Elementwise input X_I_200 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(Concatenate)]] X_T535 = add(X_T511, X_T534)
// Elementwise op: [[pid(Sub)]] X_T537 = sub(X_T535, X_I_201)
// Elementwise op: [[pid(Mul)]] X_T538 = mul(X_T537, X_I_200)
// Tile size: { 1, 4, 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 480):(376320, 13440, 480, 1):1470 KiB
// Computed true ops: 1128960
// Computed work groups: 105
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 15, 1
__kernel void kernel_c124_sdk_164(__global float* restrict  X_T535, __global float* restrict  X_T538, __global const float* restrict  X_T511, __global const float* restrict  X_T534, __global const float* restrict  X_I_201, __global const float* restrict  X_I_200)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 7; i3_lid += 1)
  {
    int i3 = ((4 * i3_lid) + i3_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((13440 * (i2_gid + i2)) + (480 * i3)) + (i4_gid + i4_tid));
      float LX_T511 = X_T511[gout_idx];
      float LX_T534 = X_T534[gout_idx];
      float LX_I_201 = X_I_201[(i4_gid + i4_tid)];
      float LX_I_200 = X_I_200[(i4_gid + i4_tid)];
      float LX_T535 = (LX_T511 + LX_T534);
      float LX_T537 = (LX_T535 - LX_I_201);
      float LX_T538 = (LX_T537 * LX_I_200);
      X_T535[gout_idx] = LX_T535;
      X_T538[gout_idx] = LX_T538;
    }
  }
}
