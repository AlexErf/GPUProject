#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 288 }
// Out stride: { 56448, 4032, 288, 1 }
// Elementwise input X_T608 shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Elementwise input X_T612 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_220 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T613 = div(X_T608, X_T612)
// Elementwise op: [[pid(Add, Switch)]] X_T614 = add(X_T613, X_I_220)
// Elementwise op: X_T615 = cmp_lt(X_T614, X_T2)
// Elementwise op: [[pid(Relu)]] X_T616 = cond(X_T615, X_T2, X_T614)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Computed true ops: 225792
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c124_sdk_191(__global float* restrict  X_T616, __global const float* restrict  X_T608, __global const float* restrict  X_T612, __global const float* restrict  X_I_220)
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
      int gout_idx = (((4032 * (i2_gid + i2_tid)) + (288 * i3)) + (i4_gid + i4_tid));
      float LX_T608 = X_T608[gout_idx];
      float LX_T612 = X_T612[(i4_gid + i4_tid)];
      float LX_I_220 = X_I_220[(i4_gid + i4_tid)];
      float LX_T613 = (LX_T608 / LX_T612);
      float LX_T614 = (LX_T613 + LX_I_220);
      int LX_T615 = (LX_T614 < 0.0f);
      float LX_T616 = select((float)LX_T614, (float)0.0f, (int)LX_T615);
      X_T616[gout_idx] = LX_T616;
    }
  }
}
