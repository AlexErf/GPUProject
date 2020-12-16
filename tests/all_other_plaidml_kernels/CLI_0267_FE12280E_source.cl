#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 16 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 512 }
// Out stride: { 100352, 7168, 512, 1 }
// Elementwise input X_T755 shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Elementwise input X_T759 shape: fp32(512):(1):2 KiB
// Elementwise input X_I_290 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T760 = div(X_T755, X_T759)
// Elementwise op: [[pid(Add, Switch)]] X_T761 = add(X_T760, X_I_290)
// Elementwise op: X_T762 = cmp_lt(X_T761, X_T2)
// Elementwise op: [[pid(Relu)]] X_T763 = cond(X_T762, X_T2, X_T761)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Computed true ops: 401408
// Computed work groups: 112
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 16, 1
__kernel void kernel_c68_sdk_254(__global float* restrict  X_T763, __global const float* restrict  X_T755, __global const float* restrict  X_T759, __global const float* restrict  X_I_290)
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
      int gout_idx = (((7168 * (i2_gid + i2_tid)) + (512 * i3)) + (i4_gid + i4_tid));
      float LX_T755 = X_T755[gout_idx];
      float LX_T759 = X_T759[(i4_gid + i4_tid)];
      float LX_I_290 = X_I_290[(i4_gid + i4_tid)];
      float LX_T760 = (LX_T755 / LX_T759);
      float LX_T761 = (LX_T760 + LX_I_290);
      int LX_T762 = (LX_T761 < 0.0f);
      float LX_T763 = select((float)LX_T761, (float)0.0f, (int)LX_T762);
      X_T763[gout_idx] = LX_T763;
    }
  }
}
