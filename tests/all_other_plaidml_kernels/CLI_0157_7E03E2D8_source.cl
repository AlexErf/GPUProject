#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Names: { i1, i2 }
// Ranges: { 1, 1000 }
// Out stride: { 1000, 1 }
// Elementwise input X_T747 shape: fp32(1, 1000):(1000, 1):3.90625 KiB
// Elementwise input X_T749 shape: fp32(1, 1):(1, 1):4 bytes
// Elementwise op: X_T723 = div(X_T747, X_T749)
// Tile size: { 1, 256 }
// Contraction output var shape: fp32(1, 1000):(1000, 1):3.90625 KiB
// Computed true ops: 1000
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c43_sdk_200(__global float* restrict  X_T723, __global const float* restrict  X_T747, __global const float* restrict  X_T749)
{
  int tid = get_local_id(0);
  int i2_gid = (get_group_id(0) * 256);
  int i2_tid = (tid % 256);
  int i2_cond = ((i2_gid != 768) || (i2_tid < 232));
  if (i2_cond)
  {
    int gout_idx = (i2_gid + i2_tid);
    float LX_T747 = X_T747[gout_idx];
    float LX_T749 = X_T749[0];
    float LX_T723 = (LX_T747 / LX_T749);
    X_T723[gout_idx] = LX_T723;
  }
}
