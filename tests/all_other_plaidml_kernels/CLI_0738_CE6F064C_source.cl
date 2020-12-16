#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1792 }
// Out stride: { 87808, 12544, 1792, 1 }
// Elementwise input X_T2503 shape: fp32(1, 7, 7, 1792):(87808, 12544, 1792, 1):343 KiB
// Elementwise input X_T2507 shape: fp32(1792):(1):7 KiB
// Elementwise input X_I_971 shape: fp32(1792):(1):7 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2508 = div(X_T2503, X_T2507)
// Elementwise op: [[pid(Add, Switch)]] X_T2509 = add(X_T2508, X_I_971)
// Elementwise op: X_T2510 = cmp_lt(X_T2509, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2511 = cond(X_T2510, X_T2, X_T2509)
// Tile size: { 1, 1, 1, 1792 }
// Contraction output var shape: fp32(1, 7, 7, 1792):(87808, 12544, 1792, 1):343 KiB
// Computed true ops: 351232
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_872(__global float* restrict  X_T2511, __global const float* restrict  X_T2503, __global const float* restrict  X_T2507, __global const float* restrict  X_I_971)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((256 * i4_lid) + i4_tid);
    int gout_idx = (((12544 * i2_gid) + (1792 * i3_gid)) + i4);
    float LX_T2503 = X_T2503[gout_idx];
    float LX_T2507 = X_T2507[i4];
    float LX_I_971 = X_I_971[i4];
    float LX_T2508 = (LX_T2503 / LX_T2507);
    float LX_T2509 = (LX_T2508 + LX_I_971);
    int LX_T2510 = (LX_T2509 < 0.0f);
    float LX_T2511 = select((float)LX_T2509, (float)0.0f, (int)LX_T2510);
    X_T2511[gout_idx] = LX_T2511;
  }
}
