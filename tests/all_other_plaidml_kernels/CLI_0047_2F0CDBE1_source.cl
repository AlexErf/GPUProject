#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28672 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 112, 112, 32 }
// Out stride: { 401408, 3584, 32, 1 }
// Elementwise input X_T88 shape: fp32(1, 112, 112, 32):(401408, 3584, 32, 1):1568 KiB
// Elementwise input X_I_105 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_104 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(Sub)]] X_T89 = sub(X_T88, X_I_105)
// Elementwise op: [[pid(Mul)]] X_T90 = mul(X_T89, X_I_104)
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
__kernel void kernel_c25_sdk_19(__global float* restrict  X_T90, __global const float* restrict  X_T88, __global const float* restrict  X_I_105, __global const float* restrict  X_I_104)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
  {
    int i2 = ((8 * i2_lid) + i2_tid);
    int gout_idx = (((3584 * i2) + (32 * i3_gid)) + i4_tid);
    float LX_T88 = X_T88[gout_idx];
    float LX_I_105 = X_I_105[i4_tid];
    float LX_I_104 = X_I_104[i4_tid];
    float LX_T89 = (LX_T88 - LX_I_105);
    float LX_T90 = (LX_T89 * LX_I_104);
    X_T90[gout_idx] = LX_T90;
  }
}
