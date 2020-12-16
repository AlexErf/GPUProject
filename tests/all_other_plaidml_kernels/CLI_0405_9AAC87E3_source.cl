#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 29 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 928 }
// Out stride: { 181888, 12992, 928, 1 }
// Elementwise input X_T1100 shape: fp32(1, 14, 14, 928):(181888, 12992, 928, 1):710.5 KiB
// Elementwise input X_T1104 shape: fp32(928):(1):3.625 KiB
// Elementwise input X_I_420 shape: fp32(928):(1):3.625 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1105 = div(X_T1100, X_T1104)
// Elementwise op: [[pid(Add, Switch)]] X_T1106 = add(X_T1105, X_I_420)
// Elementwise op: X_T1107 = cmp_lt(X_T1106, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1108 = cond(X_T1107, X_T2, X_T1106)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 928):(181888, 12992, 928, 1):710.5 KiB
// Computed true ops: 727552
// Computed work groups: 203
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 29, 1
__kernel void kernel_c108_sdk_371(__global float* restrict  X_T1108, __global const float* restrict  X_T1100, __global const float* restrict  X_T1104, __global const float* restrict  X_I_420)
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
      int gout_idx = (((12992 * (i2_gid + i2_tid)) + (928 * i3)) + (i4_gid + i4_tid));
      float LX_T1100 = X_T1100[gout_idx];
      float LX_T1104 = X_T1104[(i4_gid + i4_tid)];
      float LX_I_420 = X_I_420[(i4_gid + i4_tid)];
      float LX_T1105 = (LX_T1100 / LX_T1104);
      float LX_T1106 = (LX_T1105 + LX_I_420);
      int LX_T1107 = (LX_T1106 < 0.0f);
      float LX_T1108 = select((float)LX_T1106, (float)0.0f, (int)LX_T1107);
      X_T1108[gout_idx] = LX_T1108;
    }
  }
}
