#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1280 }
// Out stride: { 62720, 8960, 1280, 1 }
// Elementwise input X_T1895 shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
// Elementwise input X_T1899 shape: fp32(1280):(1):5 KiB
// Elementwise input X_I_731 shape: fp32(1280):(1):5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1900 = div(X_T1895, X_T1899)
// Elementwise op: [[pid(Add, Switch)]] X_T1901 = add(X_T1900, X_I_731)
// Elementwise op: X_T1902 = cmp_lt(X_T1901, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1903 = cond(X_T1902, X_T2, X_T1901)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1280):(62720, 8960, 1280, 1):245 KiB
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
__kernel void kernel_c108_sdk_656(__global float* restrict  X_T1903, __global const float* restrict  X_T1895, __global const float* restrict  X_T1899, __global const float* restrict  X_I_731)
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
      int gout_idx = (((8960 * i2_gid) + (1280 * i3_tid)) + (i4_gid + i4));
      float LX_T1895 = X_T1895[gout_idx];
      float LX_T1899 = X_T1899[(i4_gid + i4)];
      float LX_I_731 = X_I_731[(i4_gid + i4)];
      float LX_T1900 = (LX_T1895 / LX_T1899);
      float LX_T1901 = (LX_T1900 + LX_I_731);
      int LX_T1902 = (LX_T1901 < 0.0f);
      float LX_T1903 = select((float)LX_T1901, (float)0.0f, (int)LX_T1902);
      X_T1903[gout_idx] = LX_T1903;
    }
  }
}
