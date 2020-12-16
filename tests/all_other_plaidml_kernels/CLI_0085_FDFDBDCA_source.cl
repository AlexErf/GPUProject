#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4864 75 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 149, 149, 32 }
// Out stride: { 710432, 4768, 32, 1 }
// Elementwise input X_T32 shape: fp32(1, 149, 149, 32):(710432, 4768, 32, 1):2775.12 KiB
// Elementwise input X_T37 shape: fp32(32):(1):128 bytes
// Elementwise input X_I_29 shape: fp32(32):(1):128 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T38 = div(X_T32, X_T37)
// Elementwise op: [[pid(Add, Switch)]] X_T39 = add(X_T38, X_I_29)
// Elementwise op: X_T40 = cmp_lt(X_T39, X_T2)
// Elementwise op: [[pid(Relu)]] X_T41 = cond(X_T40, X_T2, X_T39)
// Tile size: { 1, 2, 8, 32 }
// Contraction output var shape: fp32(1, 149, 149, 32):(710432, 4768, 32, 1):2775.12 KiB
// Computed true ops: 2841728
// Computed work groups: 1425
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4864, 75, 1
__kernel void kernel_c51_sdk_2(__global float* restrict  X_T41, __global const float* restrict  X_T32, __global const float* restrict  X_T37, __global const float* restrict  X_I_29)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 8);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 2; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 1) || ((i3_gid != 144) || (i3_tid < 1)));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int i2_cond = ((i2_gid != 148) || (i2_tid < 1));
      if (i2_cond)
      {
        int gout_idx = (((4768 * (i2_gid + i2_tid)) + (32 * (i3_gid + i3))) + i4_tid);
        float LX_T32 = X_T32[gout_idx];
        float LX_T37 = X_T37[i4_tid];
        float LX_I_29 = X_I_29[i4_tid];
        float LX_T38 = (LX_T32 / LX_T37);
        float LX_T39 = (LX_T38 + LX_I_29);
        int LX_T40 = (LX_T39 < 0.0f);
        float LX_T41 = select((float)LX_T39, (float)0.0f, (int)LX_T40);
        X_T41[gout_idx] = LX_T41;
      }
    }
  }
}
