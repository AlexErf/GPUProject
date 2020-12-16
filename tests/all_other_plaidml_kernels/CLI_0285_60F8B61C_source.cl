#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 19 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 608 }
// Out stride: { 119168, 8512, 608, 1 }
// Elementwise input X_T830 shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Elementwise input X_T834 shape: fp32(608):(1):2.375 KiB
// Elementwise input X_I_320 shape: fp32(608):(1):2.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T835 = div(X_T830, X_T834)
// Elementwise op: [[pid(Add, Switch)]] X_T836 = add(X_T835, X_I_320)
// Elementwise op: X_T837 = cmp_lt(X_T836, X_T2)
// Elementwise op: [[pid(Relu)]] X_T838 = cond(X_T837, X_T2, X_T836)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 608):(119168, 8512, 608, 1):465.5 KiB
// Computed true ops: 476672
// Computed work groups: 133
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 19, 1
__kernel void kernel_c68_sdk_281(__global float* restrict  X_T838, __global const float* restrict  X_T830, __global const float* restrict  X_T834, __global const float* restrict  X_I_320)
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
      int gout_idx = (((8512 * (i2_gid + i2_tid)) + (608 * i3)) + (i4_gid + i4_tid));
      float LX_T830 = X_T830[gout_idx];
      float LX_T834 = X_T834[(i4_gid + i4_tid)];
      float LX_I_320 = X_I_320[(i4_gid + i4_tid)];
      float LX_T835 = (LX_T830 / LX_T834);
      float LX_T836 = (LX_T835 + LX_I_320);
      int LX_T837 = (LX_T836 < 0.0f);
      float LX_T838 = select((float)LX_T836, (float)0.0f, (int)LX_T837);
      X_T838[gout_idx] = LX_T838;
    }
  }
}
