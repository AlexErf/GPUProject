#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 128 }
// Out stride: { 401408, 7168, 128, 1 }
// Elementwise input X_T115 shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Elementwise input X_T138 shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Elementwise input X_I_50 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_49 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Concatenate)]] X_T139 = add(X_T115, X_T138)
// Elementwise op: [[pid(Sub)]] X_T141 = sub(X_T139, X_I_50)
// Elementwise op: [[pid(Mul)]] X_T142 = mul(X_T141, X_I_49)
// Tile size: { 1, 4, 1, 128 }
// Contraction output var shape: fp32(1, 56, 56, 128):(401408, 7168, 128, 1):1568 KiB
// Computed true ops: 1204224
// Computed work groups: 784
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 256
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 56, 1
__kernel void kernel_c124_sdk_23(__global float* restrict  X_T139, __global float* restrict  X_T142, __global const float* restrict  X_T115, __global const float* restrict  X_T138, __global const float* restrict  X_I_50, __global const float* restrict  X_I_49)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((7168 * (i2_gid + i2_tid)) + (128 * i3_gid)) + i4);
    float LX_T115 = X_T115[gout_idx];
    float LX_T138 = X_T138[gout_idx];
    float LX_I_50 = X_I_50[i4];
    float LX_I_49 = X_I_49[i4];
    float LX_T139 = (LX_T115 + LX_T138);
    float LX_T141 = (LX_T139 - LX_I_50);
    float LX_T142 = (LX_T141 * LX_I_49);
    X_T139[gout_idx] = LX_T139;
    X_T142[gout_idx] = LX_T142;
  }
}
