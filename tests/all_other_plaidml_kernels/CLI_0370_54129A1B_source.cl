#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 128 }
// Out stride: { 6272, 896, 128, 1 }
// Elementwise input X_T1186 shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Elementwise input X_T1190 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_447 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T1191 = div(X_T1186, X_T1190)
// Elementwise op: [[pid(Add, Switch)]] X_T1192 = add(X_T1191, X_I_447)
// Elementwise op: X_T1193 = cmp_lt(X_T1192, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1194 = cond(X_T1193, X_T2, X_T1192)
// Tile size: { 1, 1, 2, 128 }
// Contraction output var shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Computed true ops: 25088
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c68_sdk_407(__global float* restrict  X_T1194, __global const float* restrict  X_T1186, __global const float* restrict  X_T1190, __global const float* restrict  X_I_447)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 128);
  int i3_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 6) || (i3_tid < 1));
  if (i3_cond)
  {
    int gout_idx = (((896 * i2_gid) + (128 * (i3_gid + i3_tid))) + i4_tid);
    float LX_T1186 = X_T1186[gout_idx];
    float LX_T1190 = X_T1190[i4_tid];
    float LX_I_447 = X_I_447[i4_tid];
    float LX_T1191 = (LX_T1186 / LX_T1190);
    float LX_T1192 = (LX_T1191 + LX_I_447);
    int LX_T1193 = (LX_T1192 < 0.0f);
    float LX_T1194 = select((float)LX_T1192, (float)0.0f, (int)LX_T1193);
    X_T1194[gout_idx] = LX_T1194;
  }
}
