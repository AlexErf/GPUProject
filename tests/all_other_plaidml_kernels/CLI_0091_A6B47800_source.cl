#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 83, 83, 42 }
// Out stride: { 289338, 3486, 42, 1 }
// Elementwise input X_T193 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise input X_T197 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_97 shape: fp32(42):(1):168 bytes
// Elementwise input X_T173 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T198 = div(X_T193, X_T197)
// Elementwise op: [[pid(Add, Switch)]] X_T199 = add(X_T198, X_I_97)
// Elementwise op: [[pid(Add)]] X_T200 = add(X_T173, X_T199)
// Elementwise op: X_T210 = cmp_lt(X_T200, X_T1)
// Elementwise op: [[pid(Relu)]] X_T211 = cond(X_T210, X_T1, X_T200)
// Tile size: { 1, 1, 4, 42 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 1446690
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_56(__global float* restrict  X_T200, __global float* restrict  X_T211, __global const float* restrict  X_T193, __global const float* restrict  X_T197, __global const float* restrict  X_I_97, __global const float* restrict  X_T173)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 4);
  int i3_cond = ((i3_gid != 80) || (i3_tid < 3));
  if (i3_cond)
  {
    int i4_cond = (i4_tid < 42);
    if (i4_cond)
    {
      int gout_idx = (((3486 * i2_gid) + (42 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T193 = X_T193[gout_idx];
      float LX_T197 = X_T197[i4_tid];
      float LX_I_97 = X_I_97[i4_tid];
      float LX_T173 = X_T173[gout_idx];
      float LX_T198 = (LX_T193 / LX_T197);
      float LX_T199 = (LX_T198 + LX_I_97);
      float LX_T200 = (LX_T173 + LX_T199);
      int LX_T210 = (LX_T200 < 0.0f);
      float LX_T211 = select((float)LX_T200, (float)0.0f, (int)LX_T210);
      X_T200[gout_idx] = LX_T200;
      X_T211[gout_idx] = LX_T211;
    }
  }
}
