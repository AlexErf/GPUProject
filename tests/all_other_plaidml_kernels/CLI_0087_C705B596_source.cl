#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 147, 147, 32 }
// Out stride: { 691488, 4704, 32, 1 }
// Elementwise input X_T43 shape: fp32(1, 147, 147, 32):(691488, 4704, 32, 1):2701.12 KiB
// Elementwise input X_T47 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_26 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T48 = div(X_T43, X_T47)
// Elementwise op: [[pid(Add, Switch)]] X_T49 = add(X_T48, X_I_26)
// Elementwise op: X_T50 = cmp_lt(X_T49, X_T2)
// Elementwise op: [[pid(Relu)]] X_T51 = cond(X_T50, X_T2, X_T49)
// Tile size: { 1, 4, 4, 32 }
// Contraction output var shape: fp32(1, 147, 147, 32):(691488, 4704, 32, 1):2701.12 KiB
// Computed true ops: 2765952
// Computed work groups: 1369
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 37, 1
__kernel void kernel_c51_sdk_5(__global float* restrict  X_T51, __global const float* restrict  X_T43, __global const float* restrict  X_T47, __global const float* restrict  X_I_26)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 144) || (i3_tid < 3));
  if (i3_cond)
  {
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2_cond = ((i2_lid < 1) || ((i2_gid != 144) || (i2_tid < 1)));
      if (i2_cond)
      {
        int i2 = ((2 * i2_lid) + i2_tid);
        int gout_idx = (((4704 * (i2_gid + i2)) + (32 * (i3_gid + i3_tid))) + i4_tid);
        float LX_T43 = X_T43[gout_idx];
        float LX_T47 = X_T47[i4_tid];
        float LX_I_26 = X_I_26[i4_tid];
        float LX_T48 = (LX_T43 / LX_T47);
        float LX_T49 = (LX_T48 + LX_I_26);
        int LX_T50 = (LX_T49 < 0.0f);
        float LX_T51 = select((float)LX_T49, (float)0.0f, (int)LX_T50);
        X_T51[gout_idx] = LX_T51;
      }
    }
  }
}
