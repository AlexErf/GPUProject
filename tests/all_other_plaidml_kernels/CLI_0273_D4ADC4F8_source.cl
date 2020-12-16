#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 544 }
// Out stride: { 106624, 7616, 544, 1 }
// Elementwise input X_T780 shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Elementwise input X_T784 shape: fp32(544):(1):2.125 KiB
// Elementwise input X_I_300 shape: fp32(544):(1):2.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T785 = div(X_T780, X_T784)
// Elementwise op: [[pid(Add, Switch)]] X_T786 = add(X_T785, X_I_300)
// Elementwise op: X_T787 = cmp_lt(X_T786, X_T2)
// Elementwise op: [[pid(Relu)]] X_T788 = cond(X_T787, X_T2, X_T786)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Computed true ops: 426496
// Computed work groups: 119
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 17, 1
__kernel void kernel_c68_sdk_263(__global float* restrict  X_T788, __global const float* restrict  X_T780, __global const float* restrict  X_T784, __global const float* restrict  X_I_300)
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
      int gout_idx = (((7616 * (i2_gid + i2_tid)) + (544 * i3)) + (i4_gid + i4_tid));
      float LX_T780 = X_T780[gout_idx];
      float LX_T784 = X_T784[(i4_gid + i4_tid)];
      float LX_I_300 = X_I_300[(i4_gid + i4_tid)];
      float LX_T785 = (LX_T780 / LX_T784);
      float LX_T786 = (LX_T785 + LX_I_300);
      int LX_T787 = (LX_T786 < 0.0f);
      float LX_T788 = select((float)LX_T786, (float)0.0f, (int)LX_T787);
      X_T788[gout_idx] = LX_T788;
    }
  }
}
