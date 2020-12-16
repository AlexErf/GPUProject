#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1600 }
// Out stride: { 78400, 11200, 1600, 1 }
// Elementwise input X_T2145 shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Elementwise input X_T2149 shape: fp32(1600):(1):6.25 KiB
// Elementwise input X_I_831 shape: fp32(1600):(1):6.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2150 = div(X_T2145, X_T2149)
// Elementwise op: [[pid(Add, Switch)]] X_T2151 = add(X_T2150, X_I_831)
// Elementwise op: X_T2152 = cmp_lt(X_T2151, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2153 = cond(X_T2152, X_T2, X_T2151)
// Tile size: { 1, 1, 1, 1600 }
// Contraction output var shape: fp32(1, 7, 7, 1600):(78400, 11200, 1600, 1):306.25 KiB
// Computed true ops: 313600
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 600
// Computed mem write: 6400
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_746(__global float* restrict  X_T2153, __global const float* restrict  X_T2145, __global const float* restrict  X_T2149, __global const float* restrict  X_I_831)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 64));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11200 * i2_gid) + (1600 * i3_gid)) + i4);
      float LX_T2145 = X_T2145[gout_idx];
      float LX_T2149 = X_T2149[i4];
      float LX_I_831 = X_I_831[i4];
      float LX_T2150 = (LX_T2145 / LX_T2149);
      float LX_T2151 = (LX_T2150 + LX_I_831);
      int LX_T2152 = (LX_T2151 < 0.0f);
      float LX_T2153 = select((float)LX_T2151, (float)0.0f, (int)LX_T2152);
      X_T2153[gout_idx] = LX_T2153;
    }
  }
}
