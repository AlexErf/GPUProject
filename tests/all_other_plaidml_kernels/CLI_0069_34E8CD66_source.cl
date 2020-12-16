#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28416 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 111, 111, 11 }
// Out stride: { 135531, 1221, 11, 1 }
// Elementwise input X_T48 shape: fp32(1, 111, 111, 11):(135531, 1221, 11, 1):529.418 KiB
// Elementwise input X_T52 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_38 shape: fp32(11):(1):44 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T53 = div(X_T48, X_T52)
// Elementwise op: [[pid(Add, Switch)]] X_T54 = add(X_T53, X_I_38)
// Elementwise op: X_T144 = cmp_lt(X_T54, X_T1)
// Elementwise op: [[pid(Relu)]] X_T145 = cond(X_T144, X_T1, X_T54)
// Tile size: { 1, 111, 1, 11 }
// Contraction output var shape: fp32(1, 111, 111, 11):(135531, 1221, 11, 1):529.418 KiB
// Computed true ops: 542124
// Computed work groups: 111
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 1332
// Computed mem write: 28416
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28416, 1, 1
__kernel void kernel_c42_sdk_5(__global float* restrict  X_T145, __global float* restrict  X_T54, __global const float* restrict  X_T48, __global const float* restrict  X_T52, __global const float* restrict  X_I_38)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 16);
  int i2_tid = ((tid / 16) % 16);
  int i4_cond = (i4_tid < 11);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2_cond = ((i2_lid < 6) || (i2_tid < 15));
      if (i2_cond)
      {
        int i2 = ((16 * i2_lid) + i2_tid);
        int gout_idx = (((1221 * i2) + (11 * i3_gid)) + i4_tid);
        float LX_T48 = X_T48[gout_idx];
        float LX_T52 = X_T52[i4_tid];
        float LX_I_38 = X_I_38[i4_tid];
        float LX_T53 = (LX_T48 / LX_T52);
        float LX_T54 = (LX_T53 + LX_I_38);
        int LX_T144 = (LX_T54 < 0.0f);
        float LX_T145 = select((float)LX_T54, (float)0.0f, (int)LX_T144);
        X_T145[gout_idx] = LX_T145;
        X_T54[gout_idx] = LX_T54;
      }
    }
  }
}
