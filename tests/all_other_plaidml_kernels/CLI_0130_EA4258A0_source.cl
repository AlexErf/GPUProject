#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 384 }
// Out stride: { 110976, 6528, 384, 1 }
// Elementwise input X_T888 shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Elementwise input X_T892 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_11 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T893 = div(X_T888, X_T892)
// Elementwise op: [[pid(Add, Switch)]] X_T894 = add(X_T893, X_I_11)
// Elementwise op: X_T895 = cmp_lt(X_T894, X_T2)
// Elementwise op: [[pid(Relu)]] X_T896 = cond(X_T895, X_T2, X_T894)
// Tile size: { 1, 1, 1, 384 }
// Contraction output var shape: fp32(1, 17, 17, 384):(110976, 6528, 384, 1):433.5 KiB
// Computed true ops: 443904
// Computed work groups: 289
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 17, 1
__kernel void kernel_c51_sdk_290(__global float* restrict  X_T896, __global const float* restrict  X_T888, __global const float* restrict  X_T892, __global const float* restrict  X_I_11)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6528 * i2_gid) + (384 * i3_gid)) + i4);
      float LX_T888 = X_T888[gout_idx];
      float LX_T892 = X_T892[i4];
      float LX_I_11 = X_I_11[i4];
      float LX_T893 = (LX_T888 / LX_T892);
      float LX_T894 = (LX_T893 + LX_I_11);
      int LX_T895 = (LX_T894 < 0.0f);
      float LX_T896 = select((float)LX_T894, (float)0.0f, (int)LX_T895);
      X_T896[gout_idx] = LX_T896;
    }
  }
}
