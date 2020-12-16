#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14336 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 112, 112, 16 }
// Out stride: { 200704, 1792, 16, 1 }
// Elementwise input X_T86 shape: fp32(1, 112, 112, 16):(200704, 1792, 16, 1):784 KiB
// Elementwise input X_T90 shape: fp32(16):(1):64 bytes
// Elementwise input X_I_78 shape: fp32(16):(1):64 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T91 = div(X_T86, X_T90)
// Elementwise op: [[pid(Add, Switch)]] X_T92 = add(X_T91, X_I_78)
// Tile size: { 1, 112, 2, 16 }
// Contraction output var shape: fp32(1, 112, 112, 16):(200704, 1792, 16, 1):784 KiB
// Computed true ops: 401408
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 2688
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14336, 1, 1
__kernel void kernel_c43_sdk_17(__global float* restrict  X_T92, __global const float* restrict  X_T86, __global const float* restrict  X_T90, __global const float* restrict  X_I_78)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 16);
  int i3_tid = ((tid / 16) % 2);
  int i2_tid = ((tid / 32) % 8);
  for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
  {
    int i2 = ((8 * i2_lid) + i2_tid);
    int gout_idx = (((1792 * i2) + (16 * (i3_gid + i3_tid))) + i4_tid);
    float LX_T86 = X_T86[gout_idx];
    float LX_T90 = X_T90[i4_tid];
    float LX_I_78 = X_I_78[i4_tid];
    float LX_T91 = (LX_T86 / LX_T90);
    float LX_T92 = (LX_T91 + LX_I_78);
    X_T92[gout_idx] = LX_T92;
  }
}
