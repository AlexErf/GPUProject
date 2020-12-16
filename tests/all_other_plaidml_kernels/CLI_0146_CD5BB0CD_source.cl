#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 17, 17, 128 }
// Out stride: { 36992, 2176, 128, 1 }
// Elementwise input X_T951 shape: fp32(1, 17, 17, 128):(36992, 2176, 128, 1):144.5 KiB
// Elementwise input X_T955 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_348 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T956 = div(X_T951, X_T955)
// Elementwise op: [[pid(Add, Switch)]] X_T957 = add(X_T956, X_I_348)
// Elementwise op: X_T958 = cmp_lt(X_T957, X_T2)
// Elementwise op: [[pid(Relu)]] X_T959 = cond(X_T958, X_T2, X_T957)
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
__kernel void kernel_c51_sdk_312(__global float* restrict  X_T959, __global const float* restrict  X_T951, __global const float* restrict  X_T955, __global const float* restrict  X_I_348)
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
        float LX_T951 = X_T951[gout_idx];
        float LX_T955 = X_T955[i4];
        float LX_I_348 = X_I_348[i4];
        float LX_T956 = (LX_T951 / LX_T955);
        float LX_T957 = (LX_T956 + LX_I_348);
        int LX_T958 = (LX_T957 < 0.0f);
        float LX_T959 = select((float)LX_T957, (float)0.0f, (int)LX_T958);
        X_T959[gout_idx] = LX_T959;
      }
    }
  }
}
