#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 192 }
// Out stride: { 12288, 1536, 192, 1 }
// Elementwise input X_T883 shape: fp32(1, 8, 8, 192):(12288, 1536, 192, 1):48 KiB
// Elementwise input X_T887 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_297 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T888 = div(X_T883, X_T887)
// Elementwise op: [[pid(Add, Switch)]] X_T889 = add(X_T888, X_I_297)
// Elementwise op: X_T890 = cmp_lt(X_T889, X_T2)
// Elementwise op: [[pid(Relu)]] X_T891 = cond(X_T890, X_T2, X_T889)
// Tile size: { 1, 4, 1, 192 }
// Contraction output var shape: fp32(1, 8, 8, 192):(12288, 1536, 192, 1):48 KiB
// Computed true ops: 49152
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 288
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_300(__global float* restrict  X_T891, __global const float* restrict  X_T883, __global const float* restrict  X_T887, __global const float* restrict  X_I_297)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((1536 * (i2_gid + i2_tid)) + (192 * i3_gid)) + i4);
    float LX_T883 = X_T883[gout_idx];
    float LX_T887 = X_T887[i4];
    float LX_I_297 = X_I_297[i4];
    float LX_T888 = (LX_T883 / LX_T887);
    float LX_T889 = (LX_T888 + LX_I_297);
    int LX_T890 = (LX_T889 < 0.0f);
    float LX_T891 = select((float)LX_T889, (float)0.0f, (int)LX_T890);
    X_T891[gout_idx] = LX_T891;
  }
}
