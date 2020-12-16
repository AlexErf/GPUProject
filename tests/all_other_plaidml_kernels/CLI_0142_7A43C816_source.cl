#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 17 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 192 }
// Out stride: { 55488, 3264, 192, 1 }
// Elementwise input X_T935 shape: fp32(1, 17, 17, 192):(55488, 3264, 192, 1):216.75 KiB
// Elementwise input X_T939 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_338 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T940 = div(X_T935, X_T939)
// Elementwise op: [[pid(Add, Switch)]] X_T941 = add(X_T940, X_I_338)
// Elementwise op: X_T942 = cmp_lt(X_T941, X_T2)
// Elementwise op: [[pid(Relu)]] X_T943 = cond(X_T942, X_T2, X_T941)
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
__kernel void kernel_c51_sdk_308(__global float* restrict  X_T943, __global const float* restrict  X_T935, __global const float* restrict  X_T939, __global const float* restrict  X_I_338)
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
        float LX_T935 = X_T935[gout_idx];
        float LX_T939 = X_T939[i4];
        float LX_I_338 = X_I_338[i4];
        float LX_T940 = (LX_T935 / LX_T939);
        float LX_T941 = (LX_T940 + LX_I_338);
        int LX_T942 = (LX_T941 < 0.0f);
        float LX_T943 = select((float)LX_T941, (float)0.0f, (int)LX_T942);
        X_T943[gout_idx] = LX_T943;
      }
    }
  }
}
