#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 34 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1088 }
// Out stride: { 213248, 15232, 1088, 1 }
// Elementwise input X_T1233 shape: fp32(1, 14, 14, 1088):(213248, 15232, 1088, 1):833 KiB
// Elementwise input X_T1237 shape: fp32(1088):(1):4.25 KiB
// Elementwise input X_I_470 shape: fp32(1088):(1):4.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1238 = div(X_T1233, X_T1237)
// Elementwise op: [[pid(Add, Switch)]] X_T1239 = add(X_T1238, X_I_470)
// Elementwise op: X_T1240 = cmp_lt(X_T1239, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1241 = cond(X_T1240, X_T2, X_T1239)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1088):(213248, 15232, 1088, 1):833 KiB
// Computed true ops: 852992
// Computed work groups: 238
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 34, 1
__kernel void kernel_c124_sdk_416(__global float* restrict  X_T1241, __global const float* restrict  X_T1233, __global const float* restrict  X_T1237, __global const float* restrict  X_I_470)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((15232 * (i2_gid + i2_tid)) + (1088 * i3)) + (i4_gid + i4_tid));
      float LX_T1233 = X_T1233[gout_idx];
      float LX_T1237 = X_T1237[(i4_gid + i4_tid)];
      float LX_I_470 = X_I_470[(i4_gid + i4_tid)];
      float LX_T1238 = (LX_T1233 / LX_T1237);
      float LX_T1239 = (LX_T1238 + LX_I_470);
      int LX_T1240 = (LX_T1239 < 0.0f);
      float LX_T1241 = select((float)LX_T1239, (float)0.0f, (int)LX_T1240);
      X_T1241[gout_idx] = LX_T1241;
    }
  }
}
