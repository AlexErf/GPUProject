#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 35 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 96 }
// Out stride: { 117600, 3360, 96, 1 }
// Elementwise input X_T138 shape: fp32(1, 35, 35, 96):(117600, 3360, 96, 1):459.375 KiB
// Elementwise input X_T142 shape: fp32(96):(1):384 bytes
// Elementwise input X_I_71 shape: fp32(96):(1):384 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T143 = div(X_T138, X_T142)
// Elementwise op: [[pid(Add, Switch)]] X_T144 = add(X_T143, X_I_71)
// Elementwise op: X_T145 = cmp_lt(X_T144, X_T2)
// Elementwise op: [[pid(Relu)]] X_T146 = cond(X_T145, X_T2, X_T144)
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
__kernel void kernel_c56_sdk_34(__global float* restrict  X_T146, __global const float* restrict  X_T138, __global const float* restrict  X_T142, __global const float* restrict  X_I_71)
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
        float LX_T138 = X_T138[gout_idx];
        float LX_T142 = X_T142[i4];
        float LX_I_71 = X_I_71[i4];
        float LX_T143 = (LX_T138 / LX_T142);
        float LX_T144 = (LX_T143 + LX_I_71);
        int LX_T145 = (LX_T144 < 0.0f);
        float LX_T146 = select((float)LX_T144, (float)0.0f, (int)LX_T145);
        X_T146[gout_idx] = LX_T146;
      }
    }
  }
}
