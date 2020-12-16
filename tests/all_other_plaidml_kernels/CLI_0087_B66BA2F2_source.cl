#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1024 }
// Out stride: { 50176, 7168, 1024, 1 }
// Elementwise input X_T408 shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Elementwise input X_T412 shape: fp32(1024):(1):4 KiB
// Elementwise input X_I_11 shape: fp32(1024):(1):4 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T413 = div(X_T408, X_T412)
// Elementwise op: [[pid(Add, Switch)]] X_T414 = add(X_T413, X_I_11)
// Elementwise op: X_T415 = cmp_lt(X_T414, X_T10)
// Elementwise op: [[pid(Relu)]] X_T416 = cond(X_T415, X_T10, X_T414)
// Elementwise op: X_T417 = cmp_lt(X_T416, X_T9)
// Elementwise op: [[pid(Relu)]] X_T418 = cond(X_T417, X_T416, X_T9)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1024):(50176, 7168, 1024, 1):196 KiB
// Computed true ops: 301056
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c25_sdk_105(__global float* restrict  X_T418, __global const float* restrict  X_T408, __global const float* restrict  X_T412, __global const float* restrict  X_I_11)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((7168 * i2_gid) + (1024 * i3_tid)) + (i4_gid + i4));
      float LX_T408 = X_T408[gout_idx];
      float LX_T412 = X_T412[(i4_gid + i4)];
      float LX_I_11 = X_I_11[(i4_gid + i4)];
      float LX_T413 = (LX_T408 / LX_T412);
      float LX_T414 = (LX_T413 + LX_I_11);
      int LX_T415 = (LX_T414 < 0.0f);
      float LX_T416 = select((float)LX_T414, (float)0.0f, (int)LX_T415);
      int LX_T417 = (LX_T416 < 6.0f);
      float LX_T418 = select((float)6.0f, (float)LX_T416, (int)LX_T417);
      X_T418[gout_idx] = LX_T418;
    }
  }
}
