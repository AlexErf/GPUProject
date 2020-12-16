#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 192 }
// Out stride: { 37632, 2688, 192, 1 }
// Elementwise input X_T309 shape: fp32(1, 14, 14, 192):(37632, 2688, 192, 1):147 KiB
// Elementwise input X_T313 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_46 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T314 = div(X_T309, X_T313)
// Elementwise op: [[pid(Add, Switch)]] X_T315 = add(X_T314, X_I_46)
// Elementwise op: X_T316 = cmp_lt(X_T315, X_T3)
// Elementwise op: [[pid(Relu)]] X_T317 = cond(X_T316, X_T3, X_T315)
// Elementwise op: X_T318 = cmp_lt(X_T317, X_T2)
// Elementwise op: [[pid(Relu)]] X_T319 = cond(X_T318, X_T317, X_T2)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 192):(37632, 2688, 192, 1):147 KiB
// Computed true ops: 225792
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
__kernel void kernel_c43_sdk_80(__global float* restrict  X_T319, __global const float* restrict  X_T309, __global const float* restrict  X_T313, __global const float* restrict  X_I_46)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((2688 * (i2_gid + i2_tid)) + (192 * i3)) + (i4_gid + i4_tid));
      float LX_T309 = X_T309[gout_idx];
      float LX_T313 = X_T313[(i4_gid + i4_tid)];
      float LX_I_46 = X_I_46[(i4_gid + i4_tid)];
      float LX_T314 = (LX_T309 / LX_T313);
      float LX_T315 = (LX_T314 + LX_I_46);
      int LX_T316 = (LX_T315 < 0.0f);
      float LX_T317 = select((float)LX_T315, (float)0.0f, (int)LX_T316);
      int LX_T318 = (LX_T317 < 6.0f);
      float LX_T319 = select((float)6.0f, (float)LX_T317, (int)LX_T318);
      X_T319[gout_idx] = LX_T319;
    }
  }
}
