#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 960 }
// Out stride: { 47040, 6720, 960, 1 }
// Elementwise input X_T1853 shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Elementwise input X_T1857 shape: fp32(960):(1):3.75 KiB
// Elementwise input X_I_711 shape: fp32(960):(1):3.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1858 = div(X_T1853, X_T1857)
// Elementwise op: [[pid(Add, Switch)]] X_T1859 = add(X_T1858, X_I_711)
// Elementwise op: X_T1860 = cmp_lt(X_T1859, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1861 = cond(X_T1860, X_T2, X_T1859)
// Tile size: { 1, 1, 1, 960 }
// Contraction output var shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Computed true ops: 188160
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 360
// Computed mem write: 3840
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_638(__global float* restrict  X_T1861, __global const float* restrict  X_T1853, __global const float* restrict  X_T1857, __global const float* restrict  X_I_711)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_tid < 192));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((6720 * i2_gid) + (960 * i3_gid)) + i4);
      float LX_T1853 = X_T1853[gout_idx];
      float LX_T1857 = X_T1857[i4];
      float LX_I_711 = X_I_711[i4];
      float LX_T1858 = (LX_T1853 / LX_T1857);
      float LX_T1859 = (LX_T1858 + LX_I_711);
      int LX_T1860 = (LX_T1859 < 0.0f);
      float LX_T1861 = select((float)LX_T1859, (float)0.0f, (int)LX_T1860);
      X_T1861[gout_idx] = LX_T1861;
    }
  }
}
