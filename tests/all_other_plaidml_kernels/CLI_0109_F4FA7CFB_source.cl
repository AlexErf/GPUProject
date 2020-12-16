#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 64 }
// Out stride: { 200704, 3584, 64, 1 }
// Elementwise input X_T64 shape: fp32(1, 56, 56, 64):(200704, 3584, 64, 1):784 KiB
// Elementwise input X_T68 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_28 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T69 = div(X_T64, X_T68)
// Elementwise op: [[pid(Add, Switch)]] X_T70 = add(X_T69, X_I_28)
// Elementwise op: X_T71 = cmp_lt(X_T70, X_T2)
// Elementwise op: [[pid(Relu)]] X_T72 = cond(X_T71, X_T2, X_T70)
// Tile size: { 1, 4, 8, 64 }
// Contraction output var shape: fp32(1, 56, 56, 64):(200704, 3584, 64, 1):784 KiB
// Computed true ops: 802816
// Computed work groups: 98
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 8192
// Computed mem read: 768
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 14, 1
__kernel void kernel_c68_sdk_8(__global float* restrict  X_T72, __global const float* restrict  X_T64, __global const float* restrict  X_T68, __global const float* restrict  X_I_28)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 8);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i3_lid = 0; i3_lid < 2; i3_lid += 1)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
      {
        int i2 = ((2 * i2_lid) + i2_tid);
        int gout_idx = (((3584 * (i2_gid + i2)) + (64 * (i3_gid + i3))) + i4);
        float LX_T64 = X_T64[gout_idx];
        float LX_T68 = X_T68[i4];
        float LX_I_28 = X_I_28[i4];
        float LX_T69 = (LX_T64 / LX_T68);
        float LX_T70 = (LX_T69 + LX_I_28);
        int LX_T71 = (LX_T70 < 0.0f);
        float LX_T72 = select((float)LX_T70, (float)0.0f, (int)LX_T71);
        X_T72[gout_idx] = LX_T72;
      }
    }
  }
}
