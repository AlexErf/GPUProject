#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 256 }
// Out stride: { 802816, 14336, 256, 1 }
// Elementwise input X_T207 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_T230 shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Elementwise input X_I_16 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_15 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Concatenate)]] X_T231 = add(X_T207, X_T230)
// Elementwise op: [[pid(Sub)]] X_T232 = sub(X_T231, X_I_16)
// Elementwise op: [[pid(Mul)]] X_T233 = mul(X_T232, X_I_15)
// Tile size: { 1, 4, 1, 256 }
// Contraction output var shape: fp32(1, 56, 56, 256):(802816, 14336, 256, 1):3136 KiB
// Computed true ops: 2408448
// Computed work groups: 784
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 512
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 56, 1
__kernel void kernel_c108_sdk_59(__global float* restrict  X_T233, __global const float* restrict  X_T207, __global const float* restrict  X_T230, __global const float* restrict  X_I_16, __global const float* restrict  X_I_15)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((14336 * (i2_gid + i2_tid)) + (256 * i3_gid)) + i4);
    float LX_T207 = X_T207[gout_idx];
    float LX_T230 = X_T230[gout_idx];
    float LX_I_16 = X_I_16[i4];
    float LX_I_15 = X_I_15[i4];
    float LX_T231 = (LX_T207 + LX_T230);
    float LX_T232 = (LX_T231 - LX_I_16);
    float LX_T233 = (LX_T232 * LX_I_15);
    X_T233[gout_idx] = LX_T233;
  }
}
