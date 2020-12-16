#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1157376 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 1157352 }
// Out stride: { 1 }
// Elementwise input X_T98 shape: fp32(1, 83, 83, 168):(1157352, 13944, 168, 1):4520.91 KiB
// Elementwise input X_T139 shape: fp32(1, 83, 83, 168):(1157352, 13944, 168, 1):4520.91 KiB
// Elementwise op: X_T140 = add(X_T98, X_T139)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 83, 83, 168):(1157352, 13944, 168, 1):4520.91 KiB
// Computed true ops: 1157352
// Computed work groups: 4521
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1157376, 1, 1
__kernel void kernel_c42_sdk_35(__global float* restrict  X_T140, __global const float* restrict  X_T98, __global const float* restrict  X_T139)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 1157120) || (i2_i3_i4_tid < 232));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T98 = X_T98[gout_idx];
    float LX_T139 = X_T139[gout_idx];
    float LX_T140 = (LX_T98 + LX_T139);
    X_T140[gout_idx] = LX_T140;
  }
}
