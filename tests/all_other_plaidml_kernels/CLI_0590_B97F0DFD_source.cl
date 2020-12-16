#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 56 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 1792 }
// Out stride: { 351232, 25088, 1792, 1 }
// Elementwise input X_T1782 shape: fp32(1, 14, 14, 1792):(351232, 25088, 1792, 1):1372 KiB
// Elementwise input X_T1786 shape: fp32(1792):(1):7 KiB
// Elementwise input X_I_6 shape: fp32(1792):(1):7 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1787 = div(X_T1782, X_T1786)
// Elementwise op: [[pid(Add, Switch)]] X_T1788 = add(X_T1787, X_I_6)
// Elementwise op: X_T1789 = cmp_lt(X_T1788, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1790 = cond(X_T1789, X_T2, X_T1788)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 1792):(351232, 25088, 1792, 1):1372 KiB
// Computed true ops: 1404928
// Computed work groups: 392
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 56, 1
__kernel void kernel_c124_sdk_613(__global float* restrict  X_T1790, __global const float* restrict  X_T1782, __global const float* restrict  X_T1786, __global const float* restrict  X_I_6)
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
      int gout_idx = (((25088 * (i2_gid + i2_tid)) + (1792 * i3)) + (i4_gid + i4_tid));
      float LX_T1782 = X_T1782[gout_idx];
      float LX_T1786 = X_T1786[(i4_gid + i4_tid)];
      float LX_I_6 = X_I_6[(i4_gid + i4_tid)];
      float LX_T1787 = (LX_T1782 / LX_T1786);
      float LX_T1788 = (LX_T1787 + LX_I_6);
      int LX_T1789 = (LX_T1788 < 0.0f);
      float LX_T1790 = select((float)LX_T1788, (float)0.0f, (int)LX_T1789);
      X_T1790[gout_idx] = LX_T1790;
    }
  }
}
