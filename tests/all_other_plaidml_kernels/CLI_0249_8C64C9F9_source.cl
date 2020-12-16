#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 176 }
// Out stride: { 8624, 1232, 176, 1 }
// Elementwise input X_T2278 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise input X_T2282 shape: fp32(176):(1):704 bytes
// Elementwise input X_I_830 shape: fp32(176):(1):704 bytes
// Elementwise input X_T2258 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2283 = div(X_T2278, X_T2282)
// Elementwise op: [[pid(Add, Switch)]] X_T2284 = add(X_T2283, X_I_830)
// Elementwise op: [[pid(Add)]] X_T2285 = add(X_T2258, X_T2284)
// Elementwise op: X_T2295 = cmp_lt(X_T2285, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2296 = cond(X_T2295, X_T1, X_T2285)
// Tile size: { 1, 7, 1, 64 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 43120
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 224
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_872(__global float* restrict  X_T2285, __global float* restrict  X_T2296, __global const float* restrict  X_T2278, __global const float* restrict  X_T2282, __global const float* restrict  X_I_830, __global const float* restrict  X_T2258)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 64);
  int i3_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 1) || ((i4_gid != 128) || (i4_tid < 16)));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i2_cond = (i2_tid < 7);
      if (i2_cond)
      {
        int gout_idx = (((1232 * i2_tid) + (176 * i3_gid)) + (i4_gid + i4));
        float LX_T2278 = X_T2278[gout_idx];
        float LX_T2282 = X_T2282[(i4_gid + i4)];
        float LX_I_830 = X_I_830[(i4_gid + i4)];
        float LX_T2258 = X_T2258[gout_idx];
        float LX_T2283 = (LX_T2278 / LX_T2282);
        float LX_T2284 = (LX_T2283 + LX_I_830);
        float LX_T2285 = (LX_T2258 + LX_T2284);
        int LX_T2295 = (LX_T2285 < 0.0f);
        float LX_T2296 = select((float)LX_T2285, (float)0.0f, (int)LX_T2295);
        X_T2285[gout_idx] = LX_T2285;
        X_T2296[gout_idx] = LX_T2296;
      }
    }
  }
}
