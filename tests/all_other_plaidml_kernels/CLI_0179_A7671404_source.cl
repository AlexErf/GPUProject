#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 256 }
// Out stride: { 16384, 2048, 256, 1 }
// Elementwise input X_T2038 shape: fp32(1, 8, 8, 256):(16384, 2048, 256, 1):64 KiB
// Elementwise input X_T2042 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_724 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2043 = div(X_T2038, X_T2042)
// Elementwise op: [[pid(Add, Switch)]] X_T2044 = add(X_T2043, X_I_724)
// Elementwise op: X_T2045 = cmp_lt(X_T2044, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2046 = cond(X_T2045, X_T2, X_T2044)
// Tile size: { 1, 4, 1, 256 }
// Contraction output var shape: fp32(1, 8, 8, 256):(16384, 2048, 256, 1):64 KiB
// Computed true ops: 65536
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 384
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_667(__global float* restrict  X_T2046, __global const float* restrict  X_T2038, __global const float* restrict  X_T2042, __global const float* restrict  X_I_724)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((2048 * (i2_gid + i2_tid)) + (256 * i3_gid)) + i4);
    float LX_T2038 = X_T2038[gout_idx];
    float LX_T2042 = X_T2042[i4];
    float LX_I_724 = X_I_724[i4];
    float LX_T2043 = (LX_T2038 / LX_T2042);
    float LX_T2044 = (LX_T2043 + LX_I_724);
    int LX_T2045 = (LX_T2044 < 0.0f);
    float LX_T2046 = select((float)LX_T2044, (float)0.0f, (int)LX_T2045);
    X_T2046[gout_idx] = LX_T2046;
  }
}
