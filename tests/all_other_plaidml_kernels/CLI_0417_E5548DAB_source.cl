#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 864 }
// Out stride: { 169344, 12096, 864, 1 }
// Elementwise input X_T1058 shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Elementwise input X_T1062 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_400 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1063 = div(X_T1058, X_T1062)
// Elementwise op: [[pid(Add, Switch)]] X_T1064 = add(X_T1063, X_I_400)
// Elementwise op: X_T1065 = cmp_lt(X_T1064, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1066 = cond(X_T1065, X_T2, X_T1064)
// Tile size: { 1, 2, 2, 864 }
// Contraction output var shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Computed true ops: 677376
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1296
// Computed mem write: 13824
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_353(__global float* restrict  X_T1066, __global const float* restrict  X_T1058, __global const float* restrict  X_T1062, __global const float* restrict  X_I_400)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 14; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 13) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((12096 * (i2_gid + i2_tid)) + (864 * (i3_gid + i3_tid))) + i4);
      float LX_T1058 = X_T1058[gout_idx];
      float LX_T1062 = X_T1062[i4];
      float LX_I_400 = X_I_400[i4];
      float LX_T1063 = (LX_T1058 / LX_T1062);
      float LX_T1064 = (LX_T1063 + LX_I_400);
      int LX_T1065 = (LX_T1064 < 0.0f);
      float LX_T1066 = select((float)LX_T1064, (float)0.0f, (int)LX_T1065);
      X_T1066[gout_idx] = LX_T1066;
    }
  }
}
