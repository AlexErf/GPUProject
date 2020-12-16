#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1088 }
// Out stride: { 53312, 7616, 1088, 1 }
// Elementwise input X_T1953 shape: fp32(1, 7, 7, 1088):(53312, 7616, 1088, 1):208.25 KiB
// Elementwise input X_T1957 shape: fp32(1088):(1):4.25 KiB
// Elementwise input X_I_751 shape: fp32(1088):(1):4.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1958 = div(X_T1953, X_T1957)
// Elementwise op: [[pid(Add, Switch)]] X_T1959 = add(X_T1958, X_I_751)
// Elementwise op: X_T1960 = cmp_lt(X_T1959, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1961 = cond(X_T1960, X_T2, X_T1959)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1088):(53312, 7616, 1088, 1):208.25 KiB
// Computed true ops: 213248
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c124_sdk_674(__global float* restrict  X_T1961, __global const float* restrict  X_T1953, __global const float* restrict  X_T1957, __global const float* restrict  X_I_751)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1024));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((7616 * i2_gid) + (1088 * i3_tid)) + (i4_gid + i4));
        float LX_T1953 = X_T1953[gout_idx];
        float LX_T1957 = X_T1957[(i4_gid + i4)];
        float LX_I_751 = X_I_751[(i4_gid + i4)];
        float LX_T1958 = (LX_T1953 / LX_T1957);
        float LX_T1959 = (LX_T1958 + LX_I_751);
        int LX_T1960 = (LX_T1959 < 0.0f);
        float LX_T1961 = select((float)LX_T1959, (float)0.0f, (int)LX_T1960);
        X_T1961[gout_idx] = LX_T1961;
      }
    }
  }
}
