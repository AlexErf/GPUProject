#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 83, 83, 84 }
// Out stride: { 578676, 6972, 84, 1 }
// Elementwise input X_T281 shape: fp32(1, 83, 83, 84):(578676, 6972, 84, 1):2260.45 KiB
// Elementwise input X_T285 shape: fp32(84):(1):336 bytes
// Elementwise input X_I_132 shape: fp32(84):(1):336 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T286 = div(X_T281, X_T285)
// Elementwise op: [[pid(Add, Switch)]] X_T287 = add(X_T286, X_I_132)
// Elementwise op: X_T288 = cmp_lt(X_T287, X_T1)
// Elementwise op: [[pid(Relu)]] X_T289 = cond(X_T288, X_T1, X_T287)
// Tile size: { 1, 4, 1, 84 }
// Contraction output var shape: fp32(1, 83, 83, 84):(578676, 6972, 84, 1):2260.45 KiB
// Computed true ops: 2314704
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_95(__global float* restrict  X_T289, __global const float* restrict  X_T281, __global const float* restrict  X_T285, __global const float* restrict  X_I_132)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  int i2_cond = ((i2_gid != 80) || (i2_tid < 3));
  if (i2_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4_cond = ((i4_lid < 1) || (i4_tid < 20));
      if (i4_cond)
      {
        int i4 = ((64 * i4_lid) + i4_tid);
        int gout_idx = (((6972 * (i2_gid + i2_tid)) + (84 * i3_gid)) + i4);
        float LX_T281 = X_T281[gout_idx];
        float LX_T285 = X_T285[i4];
        float LX_I_132 = X_I_132[i4];
        float LX_T286 = (LX_T281 / LX_T285);
        float LX_T287 = (LX_T286 + LX_I_132);
        int LX_T288 = (LX_T287 < 0.0f);
        float LX_T289 = select((float)LX_T287, (float)0.0f, (int)LX_T288);
        X_T289[gout_idx] = LX_T289;
      }
    }
  }
}
