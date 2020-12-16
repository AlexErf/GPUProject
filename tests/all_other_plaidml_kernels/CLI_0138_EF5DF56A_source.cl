#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 314624 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 314432 }
// Out stride: { 1 }
// Elementwise input X_T897 shape: fp32(1, 17, 17, 1088):(314432, 18496, 1088, 1):1228.25 KiB
// Elementwise input X_T928 shape: fp32(1, 17, 17, 1088):(314432, 18496, 1088, 1):1228.25 KiB
// Elementwise op: X_T929 = add(X_T897, X_T928)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 17, 17, 1088):(314432, 18496, 1088, 1):1228.25 KiB
// Computed true ops: 314432
// Computed work groups: 1229
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 314624, 1, 1
__kernel void kernel_c51_sdk_302(__global float* restrict  X_T929, __global const float* restrict  X_T897, __global const float* restrict  X_T928)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 314368) || (i2_i3_i4_tid < 64));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T897 = X_T897[gout_idx];
    float LX_T928 = X_T928[gout_idx];
    float LX_T929 = (LX_T897 + LX_T928);
    X_T929[gout_idx] = LX_T929;
  }
}
