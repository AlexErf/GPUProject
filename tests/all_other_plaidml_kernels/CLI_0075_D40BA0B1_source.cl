#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28672 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 112, 112, 32 }
// Out stride: { 401408, 3584, 32, 1 }
// Elementwise input X_T71 shape: fp32(1, 112, 112, 32):(401408, 3584, 32, 1):1568 KiB
// Elementwise input X_I_84 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_83 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Sub)]] X_T72 = sub(X_T71, X_I_84)
// Elementwise op: [[pid(Mul)]] X_T73 = mul(X_T72, X_I_83)
// Tile size: { 1, 112, 1, 32 }
// Contraction output var shape: fp32(1, 112, 112, 32):(401408, 3584, 32, 1):1568 KiB
// Computed true ops: 802816
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28672, 1, 1
__kernel void kernel_c43_sdk_12(__global float* restrict  X_T73, __global const float* restrict  X_T71, __global const float* restrict  X_I_84, __global const float* restrict  X_I_83)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
  {
    int i2 = ((8 * i2_lid) + i2_tid);
    int gout_idx = (((3584 * i2) + (32 * i3_gid)) + i4_tid);
    float LX_T71 = X_T71[gout_idx];
    float LX_I_84 = X_I_84[i4_tid];
    float LX_I_83 = X_I_83[i4_tid];
    float LX_T72 = (LX_T71 - LX_I_84);
    float LX_T73 = (LX_T72 * LX_I_83);
    X_T73[gout_idx] = LX_T73;
  }
}
