#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 10 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1216 }
// Out stride: { 59584, 8512, 1216, 1 }
// Elementwise input X_T2053 shape: fp32(1, 7, 7, 1216):(59584, 8512, 1216, 1):232.75 KiB
// Elementwise input X_T2057 shape: fp32(1216):(1):4.75 KiB
// Elementwise input X_I_791 shape: fp32(1216):(1):4.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2058 = div(X_T2053, X_T2057)
// Elementwise op: [[pid(Add, Switch)]] X_T2059 = add(X_T2058, X_I_791)
// Elementwise op: X_T2060 = cmp_lt(X_T2059, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2061 = cond(X_T2060, X_T2, X_T2059)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1216):(59584, 8512, 1216, 1):232.75 KiB
// Computed true ops: 238336
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
__kernel void kernel_c124_sdk_710(__global float* restrict  X_T2061, __global const float* restrict  X_T2053, __global const float* restrict  X_T2057, __global const float* restrict  X_I_791)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1152));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((8512 * i2_gid) + (1216 * i3_tid)) + (i4_gid + i4));
        float LX_T2053 = X_T2053[gout_idx];
        float LX_T2057 = X_T2057[(i4_gid + i4)];
        float LX_I_791 = X_I_791[(i4_gid + i4)];
        float LX_T2058 = (LX_T2053 / LX_T2057);
        float LX_T2059 = (LX_T2058 + LX_I_791);
        int LX_T2060 = (LX_T2059 < 0.0f);
        float LX_T2061 = select((float)LX_T2059, (float)0.0f, (int)LX_T2060);
        X_T2061[gout_idx] = LX_T2061;
      }
    }
  }
}
