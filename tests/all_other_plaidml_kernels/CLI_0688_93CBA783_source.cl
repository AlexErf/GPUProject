#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1472 }
// Out stride: { 72128, 10304, 1472, 1 }
// Elementwise input X_T2253 shape: fp32(1, 7, 7, 1472):(72128, 10304, 1472, 1):281.75 KiB
// Elementwise input X_T2257 shape: fp32(1472):(1):5.75 KiB
// Elementwise input X_I_871 shape: fp32(1472):(1):5.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2258 = div(X_T2253, X_T2257)
// Elementwise op: [[pid(Add, Switch)]] X_T2259 = add(X_T2258, X_I_871)
// Elementwise op: X_T2260 = cmp_lt(X_T2259, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2261 = cond(X_T2260, X_T2, X_T2259)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1472):(72128, 10304, 1472, 1):281.75 KiB
// Computed true ops: 288512
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c124_sdk_782(__global float* restrict  X_T2261, __global const float* restrict  X_T2253, __global const float* restrict  X_T2257, __global const float* restrict  X_I_871)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1408));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((10304 * i2_gid) + (1472 * i3_tid)) + (i4_gid + i4));
        float LX_T2253 = X_T2253[gout_idx];
        float LX_T2257 = X_T2257[(i4_gid + i4)];
        float LX_I_871 = X_I_871[(i4_gid + i4)];
        float LX_T2258 = (LX_T2253 / LX_T2257);
        float LX_T2259 = (LX_T2258 + LX_I_871);
        int LX_T2260 = (LX_T2259 < 0.0f);
        float LX_T2261 = select((float)LX_T2259, (float)0.0f, (int)LX_T2260);
        X_T2261[gout_idx] = LX_T2261;
      }
    }
  }
}
