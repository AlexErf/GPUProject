#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 48 }
// Out stride: { 58800, 1680, 48, 1 }
// Elementwise input X_T102 shape: fp32(1, 35, 35, 48):(58800, 1680, 48, 1):229.688 KiB
// Elementwise input X_T106 shape: fp32(48):(1):192 bytes
// Elementwise input X_I_42 shape: fp32(48):(1):192 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T107 = div(X_T102, X_T106)
// Elementwise op: [[pid(Add, Switch)]] X_T108 = add(X_T107, X_I_42)
// Elementwise op: X_T109 = cmp_lt(X_T108, X_T2)
// Elementwise op: [[pid(Relu)]] X_T110 = cond(X_T109, X_T2, X_T108)
// Tile size: { 1, 1, 35, 48 }
// Contraction output var shape: fp32(1, 35, 35, 48):(58800, 1680, 48, 1):229.688 KiB
// Computed true ops: 235200
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 840
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c51_sdk_23(__global float* restrict  X_T110, __global const float* restrict  X_T102, __global const float* restrict  X_T106, __global const float* restrict  X_I_42)
{
  int tid = get_local_id(0);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || (i4_tid < 16));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
      {
        int i3_cond = ((i3_lid < 4) || (i3_tid < 3));
        if (i3_cond)
        {
          int i3 = ((8 * i3_lid) + i3_tid);
          int gout_idx = (((1680 * i2_gid) + (48 * i3)) + i4);
          float LX_T102 = X_T102[gout_idx];
          float LX_T106 = X_T106[i4];
          float LX_I_42 = X_I_42[i4];
          float LX_T107 = (LX_T102 / LX_T106);
          float LX_T108 = (LX_T107 + LX_I_42);
          int LX_T109 = (LX_T108 < 0.0f);
          float LX_T110 = select((float)LX_T108, (float)0.0f, (int)LX_T109);
          X_T110[gout_idx] = LX_T110;
        }
      }
    }
  }
}
