#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1504 }
// Out stride: { 73696, 10528, 1504, 1 }
// Elementwise input X_T2278 shape: fp32(1, 7, 7, 1504):(73696, 10528, 1504, 1):287.875 KiB
// Elementwise input X_T2282 shape: fp32(1504):(1):5.875 KiB
// Elementwise input X_I_881 shape: fp32(1504):(1):5.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2283 = div(X_T2278, X_T2282)
// Elementwise op: [[pid(Add, Switch)]] X_T2284 = add(X_T2283, X_I_881)
// Elementwise op: X_T2285 = cmp_lt(X_T2284, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2286 = cond(X_T2285, X_T2, X_T2284)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1504):(73696, 10528, 1504, 1):287.875 KiB
// Computed true ops: 294784
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c124_sdk_791(__global float* restrict  X_T2286, __global const float* restrict  X_T2278, __global const float* restrict  X_T2282, __global const float* restrict  X_I_881)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 1408));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((10528 * i2_gid) + (1504 * i3_tid)) + (i4_gid + i4));
        float LX_T2278 = X_T2278[gout_idx];
        float LX_T2282 = X_T2282[(i4_gid + i4)];
        float LX_I_881 = X_I_881[(i4_gid + i4)];
        float LX_T2283 = (LX_T2278 / LX_T2282);
        float LX_T2284 = (LX_T2283 + LX_I_881);
        int LX_T2285 = (LX_T2284 < 0.0f);
        float LX_T2286 = select((float)LX_T2284, (float)0.0f, (int)LX_T2285);
        X_T2286[gout_idx] = LX_T2286;
      }
    }
  }
}
