#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1184 }
// Out stride: { 58016, 8288, 1184, 1 }
// Elementwise input X_T1820 shape: fp32(1, 7, 7, 1184):(58016, 8288, 1184, 1):226.625 KiB
// Elementwise input X_T1824 shape: fp32(1184):(1):4.625 KiB
// Elementwise input X_I_701 shape: fp32(1184):(1):4.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1825 = div(X_T1820, X_T1824)
// Elementwise op: [[pid(Add, Switch)]] X_T1826 = add(X_T1825, X_I_701)
// Elementwise op: X_T1827 = cmp_lt(X_T1826, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1828 = cond(X_T1827, X_T2, X_T1826)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1184):(58016, 8288, 1184, 1):226.625 KiB
// Computed true ops: 232064
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
__kernel void kernel_c108_sdk_629(__global float* restrict  X_T1828, __global const float* restrict  X_T1820, __global const float* restrict  X_T1824, __global const float* restrict  X_I_701)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_gid != 1152));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((8288 * i2_gid) + (1184 * i3_tid)) + (i4_gid + i4));
        float LX_T1820 = X_T1820[gout_idx];
        float LX_T1824 = X_T1824[(i4_gid + i4)];
        float LX_I_701 = X_I_701[(i4_gid + i4)];
        float LX_T1825 = (LX_T1820 / LX_T1824);
        float LX_T1826 = (LX_T1825 + LX_I_701);
        int LX_T1827 = (LX_T1826 < 0.0f);
        float LX_T1828 = select((float)LX_T1826, (float)0.0f, (int)LX_T1827);
        X_T1828[gout_idx] = LX_T1828;
      }
    }
  }
}
