#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 256 }
// Out stride: { 50176, 3584, 256, 1 }
// Elementwise input X_T583 shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Elementwise input X_T587 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_210 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T588 = div(X_T583, X_T587)
// Elementwise op: [[pid(Add, Switch)]] X_T589 = add(X_T588, X_I_210)
// Elementwise op: X_T590 = cmp_lt(X_T589, X_T2)
// Elementwise op: [[pid(Relu)]] X_T591 = cond(X_T590, X_T2, X_T589)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 200704
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 8, 1
__kernel void kernel_c124_sdk_182(__global float* restrict  X_T591, __global const float* restrict  X_T583, __global const float* restrict  X_T587, __global const float* restrict  X_I_210)
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
      int gout_idx = (((3584 * (i2_gid + i2_tid)) + (256 * i3)) + (i4_gid + i4_tid));
      float LX_T583 = X_T583[gout_idx];
      float LX_T587 = X_T587[(i4_gid + i4_tid)];
      float LX_I_210 = X_I_210[(i4_gid + i4_tid)];
      float LX_T588 = (LX_T583 / LX_T587);
      float LX_T589 = (LX_T588 + LX_I_210);
      int LX_T590 = (LX_T589 < 0.0f);
      float LX_T591 = select((float)LX_T589, (float)0.0f, (int)LX_T590);
      X_T591[gout_idx] = LX_T591;
    }
  }
}
