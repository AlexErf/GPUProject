#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 11 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 1344 }
// Out stride: { 65856, 9408, 1344, 1 }
// Elementwise input X_T1945 shape: fp32(1, 7, 7, 1344):(65856, 9408, 1344, 1):257.25 KiB
// Elementwise input X_T1949 shape: fp32(1344):(1):5.25 KiB
// Elementwise input X_I_751 shape: fp32(1344):(1):5.25 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1950 = div(X_T1945, X_T1949)
// Elementwise op: [[pid(Add, Switch)]] X_T1951 = add(X_T1950, X_I_751)
// Elementwise op: X_T1952 = cmp_lt(X_T1951, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1953 = cond(X_T1952, X_T2, X_T1951)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 1344):(65856, 9408, 1344, 1):257.25 KiB
// Computed true ops: 263424
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 11, 1
__kernel void kernel_c108_sdk_674(__global float* restrict  X_T1953, __global const float* restrict  X_T1945, __global const float* restrict  X_T1949, __global const float* restrict  X_I_751)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 128);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 2) || (i4_gid != 1280));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((9408 * i2_gid) + (1344 * i3_tid)) + (i4_gid + i4));
        float LX_T1945 = X_T1945[gout_idx];
        float LX_T1949 = X_T1949[(i4_gid + i4)];
        float LX_I_751 = X_I_751[(i4_gid + i4)];
        float LX_T1950 = (LX_T1945 / LX_T1949);
        float LX_T1951 = (LX_T1950 + LX_I_751);
        int LX_T1952 = (LX_T1951 < 0.0f);
        float LX_T1953 = select((float)LX_T1951, (float)0.0f, (int)LX_T1952);
        X_T1953[gout_idx] = LX_T1953;
      }
    }
  }
}
