#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 73, 73, 80 }
// Out stride: { 426320, 5840, 80, 1 }
// Elementwise input X_T64 shape: fp32(1, 73, 73, 80):(426320, 5840, 80, 1):1665.31 KiB
// Elementwise input X_T68 shape: fp32(80):(1):320 bytes
// Elementwise input X_I_20 shape: fp32(80):(1):320 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T69 = div(X_T64, X_T68)
// Elementwise op: [[pid(Add, Switch)]] X_T70 = add(X_T69, X_I_20)
// Elementwise op: X_T71 = cmp_lt(X_T70, X_T2)
// Elementwise op: [[pid(Relu)]] X_T72 = cond(X_T71, X_T2, X_T70)
// Tile size: { 1, 2, 2, 80 }
// Contraction output var shape: fp32(1, 73, 73, 80):(426320, 5840, 80, 1):1665.31 KiB
// Computed true ops: 1705280
// Computed work groups: 1369
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 37, 1
__kernel void kernel_c51_sdk_12(__global float* restrict  X_T72, __global const float* restrict  X_T64, __global const float* restrict  X_T68, __global const float* restrict  X_I_20)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 72) || (i3_tid < 1));
  if (i3_cond)
  {
    int i2_cond = ((i2_gid != 72) || (i2_tid < 1));
    if (i2_cond)
    {
      for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
      {
        int i4_cond = ((i4_lid < 1) || (i4_tid < 16));
        if (i4_cond)
        {
          int i4 = ((64 * i4_lid) + i4_tid);
          int gout_idx = (((5840 * (i2_gid + i2_tid)) + (80 * (i3_gid + i3_tid))) + i4);
          float LX_T64 = X_T64[gout_idx];
          float LX_T68 = X_T68[i4];
          float LX_I_20 = X_I_20[i4];
          float LX_T69 = (LX_T64 / LX_T68);
          float LX_T70 = (LX_T69 + LX_I_20);
          int LX_T71 = (LX_T70 < 0.0f);
          float LX_T72 = select((float)LX_T70, (float)0.0f, (int)LX_T71);
          X_T72[gout_idx] = LX_T72;
        }
      }
    }
  }
}
