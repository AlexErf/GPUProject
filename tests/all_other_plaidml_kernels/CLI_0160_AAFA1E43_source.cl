#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 192 }
// Out stride: { 55488, 3264, 192, 1 }
// Elementwise input X_T385 shape: fp32(1, 17, 17, 192):(55488, 3264, 192, 1):216.75 KiB
// Elementwise input X_T389 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_23 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T390 = div(X_T385, X_T389)
// Elementwise op: [[pid(Add, Switch)]] X_T391 = add(X_T390, X_I_23)
// Elementwise op: X_T392 = cmp_lt(X_T391, X_T2)
// Elementwise op: [[pid(Relu)]] X_T393 = cond(X_T392, X_T2, X_T391)
// Tile size: { 1, 2, 1, 192 }
// Contraction output var shape: fp32(1, 17, 17, 192):(55488, 3264, 192, 1):216.75 KiB
// Computed true ops: 221952
// Computed work groups: 153
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 17, 1
__kernel void kernel_c56_sdk_130(__global float* restrict  X_T393, __global const float* restrict  X_T385, __global const float* restrict  X_T389, __global const float* restrict  X_I_23)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 128);
  int i2_tid = ((tid / 128) % 2);
  int i2_cond = ((i2_gid != 16) || (i2_tid < 1));
  if (i2_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4_cond = ((i4_lid < 1) || (i4_tid < 64));
      if (i4_cond)
      {
        int i4 = ((128 * i4_lid) + i4_tid);
        int gout_idx = (((3264 * (i2_gid + i2_tid)) + (192 * i3_gid)) + i4);
        float LX_T385 = X_T385[gout_idx];
        float LX_T389 = X_T389[i4];
        float LX_I_23 = X_I_23[i4];
        float LX_T390 = (LX_T385 / LX_T389);
        float LX_T391 = (LX_T390 + LX_I_23);
        int LX_T392 = (LX_T391 < 0.0f);
        float LX_T393 = select((float)LX_T391, (float)0.0f, (int)LX_T392);
        X_T393[gout_idx] = LX_T393;
      }
    }
  }
}
