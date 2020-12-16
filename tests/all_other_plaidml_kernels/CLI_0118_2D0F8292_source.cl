#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 22 }
// Out stride: { 17248, 616, 22, 1 }
// Elementwise input X_T294 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise input X_T298 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_115 shape: fp32(22):(1):88 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T299 = div(X_T294, X_T298)
// Elementwise op: [[pid(Add, Switch)]] X_T300 = add(X_T299, X_I_115)
// Elementwise op: X_T301 = cmp_lt(X_T300, X_T1)
// Elementwise op: [[pid(Relu)]] X_T302 = cond(X_T301, X_T1, X_T300)
// Tile size: { 1, 28, 1, 22 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 68992
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_100(__global float* restrict  X_T302, __global const float* restrict  X_T294, __global const float* restrict  X_T298, __global const float* restrict  X_I_115)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 22);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
    {
      int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
      if (i2_cond)
      {
        int i2 = ((8 * i2_lid) + i2_tid);
        int gout_idx = (((616 * i2) + (22 * i3_gid)) + i4_tid);
        float LX_T294 = X_T294[gout_idx];
        float LX_T298 = X_T298[i4_tid];
        float LX_I_115 = X_I_115[i4_tid];
        float LX_T299 = (LX_T294 / LX_T298);
        float LX_T300 = (LX_T299 + LX_I_115);
        int LX_T301 = (LX_T300 < 0.0f);
        float LX_T302 = select((float)LX_T300, (float)0.0f, (int)LX_T301);
        X_T302[gout_idx] = LX_T302;
      }
    }
  }
}
