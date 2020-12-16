#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 83, 83, 42 }
// Out stride: { 289338, 3486, 42, 1 }
// Elementwise input X_T77 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise input X_T81 shape: fp32(42):(1):168 bytes
// Elementwise input X_I_66 shape: fp32(42):(1):168 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T82 = div(X_T77, X_T81)
// Elementwise op: [[pid(Add, Switch)]] X_T83 = add(X_T82, X_I_66)
// Elementwise op: X_T84 = cmp_lt(X_T83, X_T1)
// Elementwise op: [[pid(Relu)]] X_T85 = cond(X_T84, X_T1, X_T83)
// Tile size: { 1, 1, 4, 42 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 1157352
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 168
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_14(__global float* restrict  X_T85, __global const float* restrict  X_T77, __global const float* restrict  X_T81, __global const float* restrict  X_I_66)
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
      float LX_T77 = X_T77[gout_idx];
      float LX_T81 = X_T81[i4_tid];
      float LX_I_66 = X_I_66[i4_tid];
      float LX_T82 = (LX_T77 / LX_T81);
      float LX_T83 = (LX_T82 + LX_I_66);
      int LX_T84 = (LX_T83 < 0.0f);
      float LX_T85 = select((float)LX_T83, (float)0.0f, (int)LX_T84);
      X_T85[gout_idx] = LX_T85;
    }
  }
}
