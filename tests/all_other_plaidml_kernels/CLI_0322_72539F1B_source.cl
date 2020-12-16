#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 15 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 480 }
// Out stride: { 94080, 6720, 480, 1 }
// Elementwise input X_T750 shape: fp32(1, 14, 14, 480):(94080, 6720, 480, 1):367.5 KiB
// Elementwise input X_T754 shape: fp32(480):(1):1.875 KiB
// Elementwise input X_I_280 shape: fp32(480):(1):1.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T755 = div(X_T750, X_T754)
// Elementwise op: [[pid(Add, Switch)]] X_T756 = add(X_T755, X_I_280)
// Elementwise op: X_T757 = cmp_lt(X_T756, X_T2)
// Elementwise op: [[pid(Relu)]] X_T758 = cond(X_T757, X_T2, X_T756)
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
__kernel void kernel_c108_sdk_245(__global float* restrict  X_T758, __global const float* restrict  X_T750, __global const float* restrict  X_T754, __global const float* restrict  X_I_280)
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
      float LX_T750 = X_T750[gout_idx];
      float LX_T754 = X_T754[(i4_gid + i4_tid)];
      float LX_I_280 = X_I_280[(i4_gid + i4_tid)];
      float LX_T755 = (LX_T750 / LX_T754);
      float LX_T756 = (LX_T755 + LX_I_280);
      int LX_T757 = (LX_T756 < 0.0f);
      float LX_T758 = select((float)LX_T756, (float)0.0f, (int)LX_T757);
      X_T758[gout_idx] = LX_T758;
    }
  }
}
