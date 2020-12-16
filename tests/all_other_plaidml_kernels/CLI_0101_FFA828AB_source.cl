#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 35 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 96 }
// Out stride: { 117600, 3360, 96, 1 }
// Elementwise input X_T86 shape: fp32(1, 35, 35, 96):(117600, 3360, 96, 1):459.375 KiB
// Elementwise input X_T90 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_14 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T91 = div(X_T86, X_T90)
// Elementwise op: [[pid(Add, Switch)]] X_T92 = add(X_T91, X_I_14)
// Elementwise op: X_T93 = cmp_lt(X_T92, X_T2)
// Elementwise op: [[pid(Relu)]] X_T94 = cond(X_T93, X_T2, X_T92)
// Tile size: { 1, 4, 1, 96 }
// Contraction output var shape: fp32(1, 35, 35, 96):(117600, 3360, 96, 1):459.375 KiB
// Computed true ops: 470400
// Computed work groups: 315
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 144
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 35, 1
__kernel void kernel_c51_sdk_19(__global float* restrict  X_T94, __global const float* restrict  X_T86, __global const float* restrict  X_T90, __global const float* restrict  X_I_14)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  int i2_cond = ((i2_gid != 32) || (i2_tid < 3));
  if (i2_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4_cond = ((i4_lid < 1) || (i4_tid < 32));
      if (i4_cond)
      {
        int i4 = ((64 * i4_lid) + i4_tid);
        int gout_idx = (((3360 * (i2_gid + i2_tid)) + (96 * i3_gid)) + i4);
        float LX_T86 = X_T86[gout_idx];
        float LX_T90 = X_T90[i4];
        float LX_I_14 = X_I_14[i4];
        float LX_T91 = (LX_T86 / LX_T90);
        float LX_T92 = (LX_T91 + LX_I_14);
        int LX_T93 = (LX_T92 < 0.0f);
        float LX_T94 = select((float)LX_T92, (float)0.0f, (int)LX_T93);
        X_T94[gout_idx] = LX_T94;
      }
    }
  }
}
