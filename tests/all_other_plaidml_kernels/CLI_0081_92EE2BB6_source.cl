#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 112, 112, 96 }
// Out stride: { 1204224, 10752, 96, 1 }
// Elementwise input X_T95 shape: fp32(1, 112, 112, 96):(1204224, 10752, 96, 1):4704 KiB
// Elementwise input X_T99 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_74 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T100 = div(X_T95, X_T99)
// Elementwise op: [[pid(Add, Switch)]] X_T101 = add(X_T100, X_I_74)
// Elementwise op: X_T102 = cmp_lt(X_T101, X_T3)
// Elementwise op: [[pid(Relu)]] X_T103 = cond(X_T102, X_T3, X_T101)
// Elementwise op: X_T104 = cmp_lt(X_T103, X_T2)
// Elementwise op: [[pid(Relu)]] X_T105 = cond(X_T104, X_T103, X_T2)
// Tile size: { 1, 8, 4, 96 }
// Contraction output var shape: fp32(1, 112, 112, 96):(1204224, 10752, 96, 1):4704 KiB
// Computed true ops: 7225344
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 1152
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 28, 1
__kernel void kernel_c43_sdk_20(__global float* restrict  X_T105, __global const float* restrict  X_T95, __global const float* restrict  X_T99, __global const float* restrict  X_I_74)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(1) * 4);
  int i2_gid = (get_group_id(0) * 8);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((10752 * (i2_gid + i2)) + (96 * (i3_gid + i3_tid))) + i4);
      float LX_T95 = X_T95[gout_idx];
      float LX_T99 = X_T99[i4];
      float LX_I_74 = X_I_74[i4];
      float LX_T100 = (LX_T95 / LX_T99);
      float LX_T101 = (LX_T100 + LX_I_74);
      int LX_T102 = (LX_T101 < 0.0f);
      float LX_T103 = select((float)LX_T101, (float)0.0f, (int)LX_T102);
      int LX_T104 = (LX_T103 < 6.0f);
      float LX_T105 = select((float)6.0f, (float)LX_T103, (int)LX_T104);
      X_T105[gout_idx] = LX_T105;
    }
  }
}
