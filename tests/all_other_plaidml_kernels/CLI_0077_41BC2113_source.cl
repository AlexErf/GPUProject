#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 512 }
// Out stride: { 100352, 7168, 512, 1 }
// Elementwise input X_T244 shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Elementwise input X_T248 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_59 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T249 = div(X_T244, X_T248)
// Elementwise op: [[pid(Add, Switch)]] X_T250 = add(X_T249, X_I_59)
// Elementwise op: X_T251 = cmp_lt(X_T250, X_T10)
// Elementwise op: [[pid(Relu)]] X_T252 = cond(X_T251, X_T10, X_T250)
// Elementwise op: X_T253 = cmp_lt(X_T252, X_T9)
// Elementwise op: [[pid(Relu)]] X_T254 = cond(X_T253, X_T252, X_T9)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Computed true ops: 602112
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 16, 1
__kernel void kernel_c25_sdk_62(__global float* restrict  X_T254, __global const float* restrict  X_T244, __global const float* restrict  X_T248, __global const float* restrict  X_I_59)
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
      int gout_idx = (((7168 * (i2_gid + i2_tid)) + (512 * i3)) + (i4_gid + i4_tid));
      float LX_T244 = X_T244[gout_idx];
      float LX_T248 = X_T248[(i4_gid + i4_tid)];
      float LX_I_59 = X_I_59[(i4_gid + i4_tid)];
      float LX_T249 = (LX_T244 / LX_T248);
      float LX_T250 = (LX_T249 + LX_I_59);
      int LX_T251 = (LX_T250 < 0.0f);
      float LX_T252 = select((float)LX_T250, (float)0.0f, (int)LX_T251);
      int LX_T253 = (LX_T252 < 6.0f);
      float LX_T254 = select((float)6.0f, (float)LX_T252, (int)LX_T253);
      X_T254[gout_idx] = LX_T254;
    }
  }
}
