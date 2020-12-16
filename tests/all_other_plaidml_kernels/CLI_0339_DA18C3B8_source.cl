#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 448 }
// Out stride: { 87808, 6272, 448, 1 }
// Elementwise input X_T706 shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Elementwise input X_T729 shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Elementwise input X_I_272 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_271 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T730 = add(X_T706, X_T729)
// Elementwise op: [[pid(Sub)]] X_T732 = sub(X_T730, X_I_272)
// Elementwise op: [[pid(Mul)]] X_T733 = mul(X_T732, X_I_271)
// Tile size: { 1, 2, 2, 448 }
// Contraction output var shape: fp32(1, 14, 14, 448):(87808, 6272, 448, 1):343 KiB
// Computed true ops: 263424
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_233(__global float* restrict  X_T730, __global float* restrict  X_T733, __global const float* restrict  X_T706, __global const float* restrict  X_T729, __global const float* restrict  X_I_272, __global const float* restrict  X_I_271)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((6272 * (i2_gid + i2_tid)) + (448 * (i3_gid + i3_tid))) + i4);
    float LX_T706 = X_T706[gout_idx];
    float LX_T729 = X_T729[gout_idx];
    float LX_I_272 = X_I_272[i4];
    float LX_I_271 = X_I_271[i4];
    float LX_T730 = (LX_T706 + LX_T729);
    float LX_T732 = (LX_T730 - LX_I_272);
    float LX_T733 = (LX_T732 * LX_I_271);
    X_T730[gout_idx] = LX_T730;
    X_T733[gout_idx] = LX_T733;
  }
}
