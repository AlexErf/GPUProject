#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1664 }
// Out stride: { 81536, 11648, 1664, 1 }
// Elementwise input X_T2194 shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Elementwise input X_T2198 shape: fp32(1664):(1):6.5 KiB
// Elementwise input X_I_2 shape: fp32(1664):(1):6.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2199 = div(X_T2194, X_T2198)
// Elementwise op: [[pid(Add, Switch)]] X_T2200 = add(X_T2199, X_I_2)
// Elementwise op: X_T2201 = cmp_lt(X_T2200, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2202 = cond(X_T2201, X_T2, X_T2200)
// Tile size: { 1, 1, 1, 1664 }
// Contraction output var shape: fp32(1, 7, 7, 1664):(81536, 11648, 1664, 1):318.5 KiB
// Computed true ops: 326144
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 624
// Computed mem write: 6656
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_763(__global float* restrict  X_T2202, __global const float* restrict  X_T2194, __global const float* restrict  X_T2198, __global const float* restrict  X_I_2)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 256);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 6) || (i4_tid < 128));
    if (i4_cond)
    {
      int i4 = ((256 * i4_lid) + i4_tid);
      int gout_idx = (((11648 * i2_gid) + (1664 * i3_gid)) + i4);
      float LX_T2194 = X_T2194[gout_idx];
      float LX_T2198 = X_T2198[i4];
      float LX_I_2 = X_I_2[i4];
      float LX_T2199 = (LX_T2194 / LX_T2198);
      float LX_T2200 = (LX_T2199 + LX_I_2);
      int LX_T2201 = (LX_T2200 < 0.0f);
      float LX_T2202 = select((float)LX_T2200, (float)0.0f, (int)LX_T2201);
      X_T2202[gout_idx] = LX_T2202;
    }
  }
}
