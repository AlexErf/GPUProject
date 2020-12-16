#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 207104 1 1
// lid: 256 1 1
// Names: { i2_i3_i4 }
// Ranges: { 206976 }
// Out stride: { 1 }
// Elementwise input X_T659 shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Elementwise input X_T687 shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Elementwise op: [[pid(Concatenate)]] X_T688 = add(X_T659, X_T687)
// Elementwise op: X_T689 = cmp_lt(X_T688, X_T1)
// Elementwise op: [[pid(Relu)]] X_T690 = cond(X_T689, X_T1, X_T688)
// Tile size: { 256 }
// Contraction output var shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Computed true ops: 620928
// Computed work groups: 809
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 64
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 207104, 1, 1
__kernel void kernel_c42_sdk_250(__global float* restrict  X_T690, __global const float* restrict  X_T659, __global const float* restrict  X_T687)
{
  int tid = get_local_id(0);
  int i2_i3_i4_gid = (get_group_id(0) * 256);
  int i2_i3_i4_tid = (tid % 256);
  int i2_i3_i4_cond = ((i2_i3_i4_gid != 206848) || (i2_i3_i4_tid < 128));
  if (i2_i3_i4_cond)
  {
    int gout_idx = (i2_i3_i4_gid + i2_i3_i4_tid);
    float LX_T659 = X_T659[gout_idx];
    float LX_T687 = X_T687[gout_idx];
    float LX_T688 = (LX_T659 + LX_T687);
    int LX_T689 = (LX_T688 < 0.0f);
    float LX_T690 = select((float)LX_T688, (float)0.0f, (int)LX_T689);
    X_T690[gout_idx] = LX_T690;
  }
}
