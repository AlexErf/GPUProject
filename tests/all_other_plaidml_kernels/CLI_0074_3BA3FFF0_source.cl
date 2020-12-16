#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 2048 }
// Out stride: { 100352, 14336, 2048, 1 }
// Elementwise input X_T634 shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Elementwise input X_T638 shape: fp32(2048):(1):8 KiB
// Elementwise input X_I_17 shape: fp32(2048):(1):8 KiB
// Elementwise input X_T606 shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T639 = div(X_T634, X_T638)
// Elementwise op: [[pid(Add, Switch)]] X_T640 = add(X_T639, X_I_17)
// Elementwise op: [[pid(Add)]] X_T641 = add(X_T640, X_T606)
// Elementwise op: X_T642 = cmp_lt(X_T641, X_T2)
// Elementwise op: [[pid(Relu)]] X_T643 = cond(X_T642, X_T2, X_T641)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 2048):(100352, 14336, 2048, 1):392 KiB
// Computed true ops: 501760
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 16, 1
__kernel void kernel_c29_sdk_152(__global float* restrict  X_T643, __global const float* restrict  X_T634, __global const float* restrict  X_T638, __global const float* restrict  X_I_17, __global const float* restrict  X_T606)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  int i3_cond = (i3_tid < 7);
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int gout_idx = (((14336 * i2_gid) + (2048 * i3_tid)) + (i4_gid + i4));
      float LX_T634 = X_T634[gout_idx];
      float LX_T638 = X_T638[(i4_gid + i4)];
      float LX_I_17 = X_I_17[(i4_gid + i4)];
      float LX_T606 = X_T606[gout_idx];
      float LX_T639 = (LX_T634 / LX_T638);
      float LX_T640 = (LX_T639 + LX_I_17);
      float LX_T641 = (LX_T640 + LX_T606);
      int LX_T642 = (LX_T641 < 0.0f);
      float LX_T643 = select((float)LX_T641, (float)0.0f, (int)LX_T642);
      X_T643[gout_idx] = LX_T643;
    }
  }
}
