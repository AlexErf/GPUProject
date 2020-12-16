#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 480 }
// Out stride: { 94080, 6720, 480, 1 }
// Elementwise input X_T758 shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Elementwise input X_T762 shape: fp32(480):(1):1.875 KiB
// Elementwise input X_I_280 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T763 = div(X_T758, X_T762)
// Elementwise op: [[pid(Add, Switch)]] X_T764 = add(X_T763, X_I_280)
// Elementwise op: X_T765 = cmp_lt(X_T764, X_T2)
// Elementwise op: [[pid(Relu)]] X_T766 = cond(X_T765, X_T2, X_T764)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Computed true ops: 376320
// Computed work groups: 105
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 15, 1
__kernel void kernel_c124_sdk_245(__global float* restrict  X_T766, __global const float* restrict  X_T758, __global const float* restrict  X_T762, __global const float* restrict  X_I_280)
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
      int gout_idx = (((6720 * (i2_gid + i2_tid)) + (480 * i3)) + (i4_gid + i4_tid));
      float LX_T758 = X_T758[gout_idx];
      float LX_T762 = X_T762[(i4_gid + i4_tid)];
      float LX_I_280 = X_I_280[(i4_gid + i4_tid)];
      float LX_T763 = (LX_T758 / LX_T762);
      float LX_T764 = (LX_T763 + LX_I_280);
      int LX_T765 = (LX_T764 < 0.0f);
      float LX_T766 = select((float)LX_T764, (float)0.0f, (int)LX_T765);
      X_T766[gout_idx] = LX_T766;
    }
  }
}
