#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 6144 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 49152 }
// Out stride: { 1 }
// Elementwise input X_T932 shape: fp32(1, 8, 8, 768):(49152, 6144, 768, 1):192 KiB
// Elementwise input X_T943 shape: fp32(1, 8, 8, 768):(49152, 6144, 768, 1):192 KiB
// Elementwise op: [[pid(Concatenate)]] X_T944 = add(X_T932, X_T943)
// Tile size: { 2048 }
// Contraction output var shape: fp32(1, 8, 8, 768):(49152, 6144, 768, 1):192 KiB
// Computed true ops: 49152
// Computed work groups: 24
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 512
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 6144, 1, 1
__kernel void kernel_c56_sdk_321(__global float* restrict  X_T944, __global const float* restrict  X_T932, __global const float* restrict  X_T943)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 2048);
  int i2_i3_i4_tid = (tid % 256);
  for (int i2_i3_i4_lid = 0; i2_i3_i4_lid < 8; i2_i3_i4_lid += 1)
  {
    int i2_i3_i4 = ((256 * i2_i3_i4_lid) + i2_i3_i4_tid);
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4);
    float LX_T932 = X_T932[gout_idx];
    float LX_T943 = X_T943[gout_idx];
    float LX_T944 = (LX_T932 + LX_T943);
    X_T944[gout_idx] = LX_T944;
  }
}
