#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 128 }
// Out stride: { 36992, 2176, 128, 1 }
// Elementwise input X_T401 shape: fp32(1, 17, 17, 128):(36992, 2176, 128, 1):144.5 KiB
// Elementwise input X_T405 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_154 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T406 = div(X_T401, X_T405)
// Elementwise op: [[pid(Add, Switch)]] X_T407 = add(X_T406, X_I_154)
// Elementwise op: X_T408 = cmp_lt(X_T407, X_T2)
// Elementwise op: [[pid(Relu)]] X_T409 = cond(X_T408, X_T2, X_T407)
// Tile size: { 1, 1, 17, 128 }
// Contraction output var shape: fp32(1, 17, 17, 128):(36992, 2176, 128, 1):144.5 KiB
// Computed true ops: 147968
// Computed work groups: 17
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 816
// Computed mem write: 8704
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 1, 1
__kernel void kernel_c56_sdk_134(__global float* restrict  X_T409, __global const float* restrict  X_T401, __global const float* restrict  X_T405, __global const float* restrict  X_I_154)
{
  int tid = get_local_id(0);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i3_lid = 0; i3_lid < 3; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 2) || (i3_tid < 1));
    if (i3_cond)
    {
      int i3 = ((8 * i3_lid) + i3_tid);
      for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
      {
        int i4 = ((32 * i4_lid) + i4_tid);
        int gout_idx = (((2176 * i2_gid) + (128 * i3)) + i4);
        float LX_T401 = X_T401[gout_idx];
        float LX_T405 = X_T405[i4];
        float LX_I_154 = X_I_154[i4];
        float LX_T406 = (LX_T401 / LX_T405);
        float LX_T407 = (LX_T406 + LX_I_154);
        int LX_T408 = (LX_T407 < 0.0f);
        float LX_T409 = select((float)LX_T407, (float)0.0f, (int)LX_T408);
        X_T409[gout_idx] = LX_T409;
      }
    }
  }
}
