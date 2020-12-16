#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1152 }
// Out stride: { 56448, 8064, 1152, 1 }
// Elementwise input X_T2003 shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
// Elementwise input X_T2007 shape: fp32(1152):(1):4.5 KiB
// Elementwise input X_I_771 shape: fp32(1152):(1):4.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2008 = div(X_T2003, X_T2007)
// Elementwise op: [[pid(Add, Switch)]] X_T2009 = add(X_T2008, X_I_771)
// Elementwise op: X_T2010 = cmp_lt(X_T2009, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2011 = cond(X_T2010, X_T2, X_T2009)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1152):(56448, 8064, 1152, 1):220.5 KiB
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
__kernel void kernel_c124_sdk_692(__global float* restrict  X_T2011, __global const float* restrict  X_T2003, __global const float* restrict  X_T2007, __global const float* restrict  X_I_771)
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
      int gout_idx = (((8064 * i2_gid) + (1152 * i3_tid)) + (i4_gid + i4));
      float LX_T2003 = X_T2003[gout_idx];
      float LX_T2007 = X_T2007[(i4_gid + i4)];
      float LX_I_771 = X_I_771[(i4_gid + i4)];
      float LX_T2008 = (LX_T2003 / LX_T2007);
      float LX_T2009 = (LX_T2008 + LX_I_771);
      int LX_T2010 = (LX_T2009 < 0.0f);
      float LX_T2011 = select((float)LX_T2009, (float)0.0f, (int)LX_T2010);
      X_T2011[gout_idx] = LX_T2011;
    }
  }
}
