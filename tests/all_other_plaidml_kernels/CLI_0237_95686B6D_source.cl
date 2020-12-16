#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 320 }
// Out stride: { 62720, 4480, 320, 1 }
// Elementwise input X_T605 shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Elementwise input X_T609 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_230 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T610 = div(X_T605, X_T609)
// Elementwise op: [[pid(Add, Switch)]] X_T611 = add(X_T610, X_I_230)
// Elementwise op: X_T612 = cmp_lt(X_T611, X_T2)
// Elementwise op: [[pid(Relu)]] X_T613 = cond(X_T612, X_T2, X_T611)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 320):(62720, 4480, 320, 1):245 KiB
// Computed true ops: 250880
// Computed work groups: 70
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 10, 1
__kernel void kernel_c68_sdk_200(__global float* restrict  X_T613, __global const float* restrict  X_T605, __global const float* restrict  X_T609, __global const float* restrict  X_I_230)
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
      int gout_idx = (((4480 * (i2_gid + i2_tid)) + (320 * i3)) + (i4_gid + i4_tid));
      float LX_T605 = X_T605[gout_idx];
      float LX_T609 = X_T609[(i4_gid + i4_tid)];
      float LX_I_230 = X_I_230[(i4_gid + i4_tid)];
      float LX_T610 = (LX_T605 / LX_T609);
      float LX_T611 = (LX_T610 + LX_I_230);
      int LX_T612 = (LX_T611 < 0.0f);
      float LX_T613 = select((float)LX_T611, (float)0.0f, (int)LX_T612);
      X_T613[gout_idx] = LX_T613;
    }
  }
}
