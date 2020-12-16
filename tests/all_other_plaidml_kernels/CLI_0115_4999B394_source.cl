#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 64 }
// Out stride: { 78400, 2240, 64, 1 }
// Elementwise input X_T86 shape: fp32(1, 35, 35, 64):(78400, 2240, 64, 1):306.25 KiB
// Elementwise input X_T90 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_35 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T91 = div(X_T86, X_T90)
// Elementwise op: [[pid(Add, Switch)]] X_T92 = add(X_T91, X_I_35)
// Elementwise op: X_T93 = cmp_lt(X_T92, X_T2)
// Elementwise op: [[pid(Relu)]] X_T94 = cond(X_T93, X_T2, X_T92)
// Tile size: { 1, 1, 35, 64 }
// Contraction output var shape: fp32(1, 35, 35, 64):(78400, 2240, 64, 1):306.25 KiB
// Computed true ops: 313600
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 840
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c56_sdk_19(__global float* restrict  X_T94, __global const float* restrict  X_T86, __global const float* restrict  X_T90, __global const float* restrict  X_I_35)
{
  int tid = get_local_id(0);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 4) || (i3_tid < 3));
    if (i3_cond)
    {
      int i3 = ((8 * i3_lid) + i3_tid);
      for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
      {
        int i4 = ((32 * i4_lid) + i4_tid);
        int gout_idx = (((2240 * i2_gid) + (64 * i3)) + i4);
        float LX_T86 = X_T86[gout_idx];
        float LX_T90 = X_T90[i4];
        float LX_I_35 = X_I_35[i4];
        float LX_T91 = (LX_T86 / LX_T90);
        float LX_T92 = (LX_T91 + LX_I_35);
        int LX_T93 = (LX_T92 < 0.0f);
        float LX_T94 = select((float)LX_T92, (float)0.0f, (int)LX_T93);
        X_T94[gout_idx] = LX_T94;
      }
    }
  }
}
