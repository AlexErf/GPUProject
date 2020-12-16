#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 21248 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 165, 165, 42 }
// Out stride: { 1143450, 6930, 42, 1 }
// Elementwise input X_T51 shape: fp32(1, 165, 165, 42):(1143450, 6930, 42, 1):4466.6 KiB
// Elementwise input X_T55 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_50 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T56 = div(X_T51, X_T55)
// Elementwise op: [[pid(Add, Switch)]] X_T57 = add(X_T56, X_I_50)
// Elementwise op: X_T147 = cmp_lt(X_T57, X_T1)
// Elementwise op: [[pid(Relu)]] X_T148 = cond(X_T147, X_T1, X_T57)
// Tile size: { 1, 2, 2, 42 }
// Contraction output var shape: fp32(1, 165, 165, 42):(1143450, 6930, 42, 1):4466.6 KiB
// Computed true ops: 4573800
// Computed work groups: 6889
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 2048
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 21248, 83, 1
__kernel void kernel_c42_sdk_5(__global float* restrict  X_T148, __global float* restrict  X_T57, __global const float* restrict  X_T51, __global const float* restrict  X_T55, __global const float* restrict  X_I_50)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 164) || (i3_tid < 1));
  if (i3_cond)
  {
    int i2_cond = ((i2_gid != 164) || (i2_tid < 1));
    if (i2_cond)
    {
      int i4_cond = (i4_tid < 42);
      if (i4_cond)
      {
        int gout_idx = (((6930 * (i2_gid + i2_tid)) + (42 * (i3_gid + i3_tid))) + i4_tid);
        float LX_T51 = X_T51[gout_idx];
        float LX_T55 = X_T55[i4_tid];
        float LX_I_50 = X_I_50[i4_tid];
        float LX_T56 = (LX_T51 / LX_T55);
        float LX_T57 = (LX_T56 + LX_I_50);
        int LX_T147 = (LX_T57 < 0.0f);
        float LX_T148 = select((float)LX_T57, (float)0.0f, (int)LX_T147);
        X_T148[gout_idx] = LX_T148;
        X_T57[gout_idx] = LX_T57;
      }
    }
  }
}
