#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 544 }
// Out stride: { 26656, 3808, 544, 1 }
// Elementwise input X_T1200 shape: fp32(1, 7, 7, 544):(26656, 3808, 544, 1):104.125 KiB
// Elementwise input X_T1204 shape: fp32(544):(1):2.125 KiB
// Elementwise input X_I_461 shape: fp32(544):(1):2.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1205 = div(X_T1200, X_T1204)
// Elementwise op: [[pid(Add, Switch)]] X_T1206 = add(X_T1205, X_I_461)
// Elementwise op: X_T1207 = cmp_lt(X_T1206, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1208 = cond(X_T1207, X_T2, X_T1206)
// Tile size: { 1, 1, 1, 544 }
// Contraction output var shape: fp32(1, 7, 7, 544):(26656, 3808, 544, 1):104.125 KiB
// Computed true ops: 106624
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 204
// Computed mem write: 2176
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_413(__global float* restrict  X_T1208, __global const float* restrict  X_T1200, __global const float* restrict  X_T1204, __global const float* restrict  X_I_461)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((3808 * i2_gid) + (544 * i3_gid)) + i4);
      float LX_T1200 = X_T1200[gout_idx];
      float LX_T1204 = X_T1204[i4];
      float LX_I_461 = X_I_461[i4];
      float LX_T1205 = (LX_T1200 / LX_T1204);
      float LX_T1206 = (LX_T1205 + LX_I_461);
      int LX_T1207 = (LX_T1206 < 0.0f);
      float LX_T1208 = select((float)LX_T1206, (float)0.0f, (int)LX_T1207);
      X_T1208[gout_idx] = LX_T1208;
    }
  }
}
