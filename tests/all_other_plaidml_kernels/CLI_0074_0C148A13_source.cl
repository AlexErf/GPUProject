#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 256 }
// Out stride: { 50176, 3584, 256, 1 }
// Elementwise input X_T231 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_T235 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_63 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T236 = div(X_T231, X_T235)
// Elementwise op: [[pid(Add, Switch)]] X_T237 = add(X_T236, X_I_63)
// Elementwise op: X_T238 = cmp_lt(X_T237, X_T10)
// Elementwise op: [[pid(Relu)]] X_T239 = cond(X_T238, X_T10, X_T237)
// Elementwise op: X_T240 = cmp_lt(X_T239, X_T9)
// Elementwise op: [[pid(Relu)]] X_T241 = cond(X_T240, X_T239, X_T9)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
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
__kernel void kernel_c25_sdk_59(__global float* restrict  X_T241, __global const float* restrict  X_T231, __global const float* restrict  X_T235, __global const float* restrict  X_I_63)
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
      float LX_T231 = X_T231[gout_idx];
      float LX_T235 = X_T235[(i4_gid + i4_tid)];
      float LX_I_63 = X_I_63[(i4_gid + i4_tid)];
      float LX_T236 = (LX_T231 / LX_T235);
      float LX_T237 = (LX_T236 + LX_I_63);
      int LX_T238 = (LX_T237 < 0.0f);
      float LX_T239 = select((float)LX_T237, (float)0.0f, (int)LX_T238);
      int LX_T240 = (LX_T239 < 6.0f);
      float LX_T241 = select((float)6.0f, (float)LX_T239, (int)LX_T240);
      X_T241[gout_idx] = LX_T241;
    }
  }
}
