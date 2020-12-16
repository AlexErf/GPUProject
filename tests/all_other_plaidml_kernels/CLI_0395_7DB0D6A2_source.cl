#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 672 }
// Out stride: { 32928, 4704, 672, 1 }
// Elementwise input X_T1300 shape: fp32(1, 7, 7, 672):(32928, 4704, 672, 1):128.625 KiB
// Elementwise input X_T1304 shape: fp32(672):(1):2.625 KiB
// Elementwise input X_I_501 shape: fp32(672):(1):2.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1305 = div(X_T1300, X_T1304)
// Elementwise op: [[pid(Add, Switch)]] X_T1306 = add(X_T1305, X_I_501)
// Elementwise op: X_T1307 = cmp_lt(X_T1306, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1308 = cond(X_T1307, X_T2, X_T1306)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 672):(32928, 4704, 672, 1):128.625 KiB
// Computed true ops: 131712
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c68_sdk_449(__global float* restrict  X_T1308, __global const float* restrict  X_T1300, __global const float* restrict  X_T1304, __global const float* restrict  X_I_501)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 640));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((4704 * i2_gid) + (672 * i3_tid)) + (i4_gid + i4));
        float LX_T1300 = X_T1300[gout_idx];
        float LX_T1304 = X_T1304[(i4_gid + i4)];
        float LX_I_501 = X_I_501[(i4_gid + i4)];
        float LX_T1305 = (LX_T1300 / LX_T1304);
        float LX_T1306 = (LX_T1305 + LX_I_501);
        int LX_T1307 = (LX_T1306 < 0.0f);
        float LX_T1308 = select((float)LX_T1306, (float)0.0f, (int)LX_T1307);
        X_T1308[gout_idx] = LX_T1308;
      }
    }
  }
}
