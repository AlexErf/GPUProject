#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1280 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 160 }
// Out stride: { 46240, 2720, 160, 1 }
// Elementwise input X_T516 shape: fp32(1, 17, 17, 160):(46240, 2720, 160, 1):180.625 KiB
// Elementwise input X_T520 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_191 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T521 = div(X_T516, X_T520)
// Elementwise op: [[pid(Add, Switch)]] X_T522 = add(X_T521, X_I_191)
// Elementwise op: X_T523 = cmp_lt(X_T522, X_T2)
// Elementwise op: [[pid(Relu)]] X_T524 = cond(X_T523, X_T2, X_T522)
// Tile size: { 1, 2, 17, 32 }
// Contraction output var shape: fp32(1, 17, 17, 160):(46240, 2720, 160, 1):180.625 KiB
// Computed true ops: 184960
// Computed work groups: 45
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 408
// Computed mem write: 4352
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1280, 9, 1
__kernel void kernel_c56_sdk_174(__global float* restrict  X_T524, __global const float* restrict  X_T516, __global const float* restrict  X_T520, __global const float* restrict  X_I_191)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  int i2_cond = ((i2_gid != 16) || (i2_tid < 1));
  if (i2_cond)
  {
    for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
    {
      int i3_cond = ((i3_lid < 4) || (i3_tid < 1));
      if (i3_cond)
      {
        int i3 = ((4 * i3_lid) + i3_tid);
        int gout_idx = (((2720 * (i2_gid + i2_tid)) + (160 * i3)) + (i4_gid + i4_tid));
        float LX_T516 = X_T516[gout_idx];
        float LX_T520 = X_T520[(i4_gid + i4_tid)];
        float LX_I_191 = X_I_191[(i4_gid + i4_tid)];
        float LX_T521 = (LX_T516 / LX_T520);
        float LX_T522 = (LX_T521 + LX_I_191);
        int LX_T523 = (LX_T522 < 0.0f);
        float LX_T524 = select((float)LX_T522, (float)0.0f, (int)LX_T523);
        X_T524[gout_idx] = LX_T524;
      }
    }
  }
}
