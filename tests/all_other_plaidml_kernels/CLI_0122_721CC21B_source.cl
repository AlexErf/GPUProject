#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 64 }
// Out stride: { 12544, 896, 64, 1 }
// Elementwise input X_T359 shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Elementwise input X_T363 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_148 shape: fp32(64):(1):256 bytes
// Elementwise input X_T328 shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T364 = div(X_T359, X_T363)
// Elementwise op: [[pid(Add, Switch)]] X_T365 = add(X_T364, X_I_148)
// Elementwise op: [[pid(Add)]] X_T366 = add(X_T328, X_T365)
// Tile size: { 1, 2, 2, 64 }
// Contraction output var shape: fp32(1, 14, 14, 64):(12544, 896, 64, 1):49 KiB
// Computed true ops: 37632
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c43_sdk_94(__global float* restrict  X_T366, __global const float* restrict  X_T359, __global const float* restrict  X_T363, __global const float* restrict  X_I_148, __global const float* restrict  X_T328)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int gout_idx = (((896 * (i2_gid + i2_tid)) + (64 * (i3_gid + i3_tid))) + i4_tid);
  float LX_T359 = X_T359[gout_idx];
  float LX_T363 = X_T363[i4_tid];
  float LX_I_148 = X_I_148[i4_tid];
  float LX_T328 = X_T328[gout_idx];
  float LX_T364 = (LX_T359 / LX_T363);
  float LX_T365 = (LX_T364 + LX_I_148);
  float LX_T366 = (LX_T328 + LX_T365);
  X_T366[gout_idx] = LX_T366;
}
