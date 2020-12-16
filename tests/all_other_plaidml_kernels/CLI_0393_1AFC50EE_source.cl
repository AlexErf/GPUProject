#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 864 }
// Out stride: { 169344, 12096, 864, 1 }
// Elementwise input X_T1050 shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Elementwise input X_T1054 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_400 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1055 = div(X_T1050, X_T1054)
// Elementwise op: [[pid(Add, Switch)]] X_T1056 = add(X_T1055, X_I_400)
// Elementwise op: X_T1057 = cmp_lt(X_T1056, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1058 = cond(X_T1057, X_T2, X_T1056)
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
__kernel void kernel_c108_sdk_353(__global float* restrict  X_T1058, __global const float* restrict  X_T1050, __global const float* restrict  X_T1054, __global const float* restrict  X_I_400)
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
      float LX_T1050 = X_T1050[gout_idx];
      float LX_T1054 = X_T1054[i4];
      float LX_I_400 = X_I_400[i4];
      float LX_T1055 = (LX_T1050 / LX_T1054);
      float LX_T1056 = (LX_T1055 + LX_I_400);
      int LX_T1057 = (LX_T1056 < 0.0f);
      float LX_T1058 = select((float)LX_T1056, (float)0.0f, (int)LX_T1057);
      X_T1058[gout_idx] = LX_T1058;
    }
  }
}
