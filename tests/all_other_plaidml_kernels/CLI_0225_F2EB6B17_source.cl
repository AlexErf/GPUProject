#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 256 }
// Out stride: { 50176, 3584, 256, 1 }
// Elementwise input X_T555 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_T559 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_210 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T560 = div(X_T555, X_T559)
// Elementwise op: [[pid(Add, Switch)]] X_T561 = add(X_T560, X_I_210)
// Elementwise op: X_T562 = cmp_lt(X_T561, X_T2)
// Elementwise op: [[pid(Relu)]] X_T563 = cond(X_T562, X_T2, X_T561)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 200704
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
__kernel void kernel_c68_sdk_182(__global float* restrict  X_T563, __global const float* restrict  X_T555, __global const float* restrict  X_T559, __global const float* restrict  X_I_210)
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
      int gout_idx = (((3584 * (i2_gid + i2_tid)) + (256 * i3)) + (i4_gid + i4_tid));
      float LX_T555 = X_T555[gout_idx];
      float LX_T559 = X_T559[(i4_gid + i4_tid)];
      float LX_I_210 = X_I_210[(i4_gid + i4_tid)];
      float LX_T560 = (LX_T555 / LX_T559);
      float LX_T561 = (LX_T560 + LX_I_210);
      int LX_T562 = (LX_T561 < 0.0f);
      float LX_T563 = select((float)LX_T561, (float)0.0f, (int)LX_T562);
      X_T563[gout_idx] = LX_T563;
    }
  }
}
