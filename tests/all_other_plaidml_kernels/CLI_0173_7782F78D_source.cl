#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 192 }
// Out stride: { 12288, 1536, 192, 1 }
// Elementwise input X_T2003 shape: fp32(1, 8, 8, 192):(12288, 1536, 192, 1):48 KiB
// Elementwise input X_T2007 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_720 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T2008 = div(X_T2003, X_T2007)
// Elementwise op: [[pid(Add, Switch)]] X_T2009 = add(X_T2008, X_I_720)
// Elementwise op: X_T2010 = cmp_lt(X_T2009, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2011 = cond(X_T2010, X_T2, X_T2009)
// Tile size: { 1, 4, 1, 192 }
// Contraction output var shape: fp32(1, 8, 8, 192):(12288, 1536, 192, 1):48 KiB
// Computed true ops: 49152
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 3072
// Computed mem read: 288
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_657(__global float* restrict  X_T2011, __global const float* restrict  X_T2003, __global const float* restrict  X_T2007, __global const float* restrict  X_I_720)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 3; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((1536 * (i2_gid + i2_tid)) + (192 * i3_gid)) + i4);
    float LX_T2003 = X_T2003[gout_idx];
    float LX_T2007 = X_T2007[i4];
    float LX_I_720 = X_I_720[i4];
    float LX_T2008 = (LX_T2003 / LX_T2007);
    float LX_T2009 = (LX_T2008 + LX_I_720);
    int LX_T2010 = (LX_T2009 < 0.0f);
    float LX_T2011 = select((float)LX_T2009, (float)0.0f, (int)LX_T2010);
    X_T2011[gout_idx] = LX_T2011;
  }
}
