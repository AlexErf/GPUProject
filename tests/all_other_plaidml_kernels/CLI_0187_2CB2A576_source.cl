#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 320 }
// Out stride: { 20480, 2560, 320, 1 }
// Elementwise input X_T842 shape: fp32(1, 8, 8, 320):(20480, 2560, 320, 1):80 KiB
// Elementwise input X_T846 shape: fp32(320):(1):1.25 KiB
// Elementwise input X_I_8 shape: fp32(320):(1):1.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T847 = div(X_T842, X_T846)
// Elementwise op: [[pid(Add, Switch)]] X_T848 = add(X_T847, X_I_8)
// Elementwise op: X_T849 = cmp_lt(X_T848, X_T2)
// Elementwise op: [[pid(Relu)]] X_T850 = cond(X_T849, X_T2, X_T848)
// Tile size: { 1, 4, 1, 320 }
// Contraction output var shape: fp32(1, 8, 8, 320):(20480, 2560, 320, 1):80 KiB
// Computed true ops: 81920
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 480
// Computed mem write: 5120
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_287(__global float* restrict  X_T850, __global const float* restrict  X_T842, __global const float* restrict  X_T846, __global const float* restrict  X_I_8)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((2560 * (i2_gid + i2_tid)) + (320 * i3_gid)) + i4);
    float LX_T842 = X_T842[gout_idx];
    float LX_T846 = X_T846[i4];
    float LX_I_8 = X_I_8[i4];
    float LX_T847 = (LX_T842 / LX_T846);
    float LX_T848 = (LX_T847 + LX_I_8);
    int LX_T849 = (LX_T848 < 0.0f);
    float LX_T850 = select((float)LX_T848, (float)0.0f, (int)LX_T849);
    X_T850[gout_idx] = LX_T850;
  }
}
