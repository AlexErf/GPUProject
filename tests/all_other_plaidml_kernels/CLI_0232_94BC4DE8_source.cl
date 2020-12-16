#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 176 }
// Out stride: { 34496, 2464, 176, 1 }
// Elementwise input X_T2150 shape: fp32(1, 14, 14, 176):(34496, 2464, 176, 1):134.75 KiB
// Elementwise input X_T2154 shape: fp32(176):(1):704 bytes
// Elementwise input X_I_799 shape: fp32(176):(1):704 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T2155 = div(X_T2150, X_T2154)
// Elementwise op: [[pid(Add, Switch)]] X_T2156 = add(X_T2155, X_I_799)
// Elementwise op: X_T2157 = cmp_lt(X_T2156, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2158 = cond(X_T2157, X_T1, X_T2156)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 176):(34496, 2464, 176, 1):134.75 KiB
// Computed true ops: 137984
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c42_sdk_825(__global float* restrict  X_T2158, __global const float* restrict  X_T2150, __global const float* restrict  X_T2154, __global const float* restrict  X_I_799)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i4_cond = ((i4_gid != 160) || (i4_tid < 16));
  if (i4_cond)
  {
    for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
    {
      int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
      if (i3_cond)
      {
        int i3 = ((4 * i3_lid) + i3_tid);
        int gout_idx = (((2464 * (i2_gid + i2_tid)) + (176 * i3)) + (i4_gid + i4_tid));
        float LX_T2150 = X_T2150[gout_idx];
        float LX_T2154 = X_T2154[(i4_gid + i4_tid)];
        float LX_I_799 = X_I_799[(i4_gid + i4_tid)];
        float LX_T2155 = (LX_T2150 / LX_T2154);
        float LX_T2156 = (LX_T2155 + LX_I_799);
        int LX_T2157 = (LX_T2156 < 0.0f);
        float LX_T2158 = select((float)LX_T2156, (float)0.0f, (int)LX_T2157);
        X_T2158[gout_idx] = LX_T2158;
      }
    }
  }
}
