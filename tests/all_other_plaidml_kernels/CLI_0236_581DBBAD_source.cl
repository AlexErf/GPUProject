#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 176 }
// Out stride: { 8624, 1232, 176, 1 }
// Elementwise input X_T2165 shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Elementwise input X_T2169 shape: fp32(176):(1):704 bytes
// Elementwise input X_I_794 shape: fp32(176):(1):704 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T2170 = div(X_T2165, X_T2169)
// Elementwise op: [[pid(Add, Switch)]] X_T2171 = add(X_T2170, X_I_794)
// Elementwise op: X_T2172 = cmp_lt(X_T2171, X_T1)
// Elementwise op: [[pid(Relu)]] X_T2173 = cond(X_T2172, X_T1, X_T2171)
// Tile size: { 1, 7, 1, 64 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 34496
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 168
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_830(__global float* restrict  X_T2173, __global const float* restrict  X_T2165, __global const float* restrict  X_T2169, __global const float* restrict  X_I_794)
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
        float LX_T2165 = X_T2165[gout_idx];
        float LX_T2169 = X_T2169[(i4_gid + i4)];
        float LX_I_794 = X_I_794[(i4_gid + i4)];
        float LX_T2170 = (LX_T2165 / LX_T2169);
        float LX_T2171 = (LX_T2170 + LX_I_794);
        int LX_T2172 = (LX_T2171 < 0.0f);
        float LX_T2173 = select((float)LX_T2171, (float)0.0f, (int)LX_T2172);
        X_T2173[gout_idx] = LX_T2173;
      }
    }
  }
}
