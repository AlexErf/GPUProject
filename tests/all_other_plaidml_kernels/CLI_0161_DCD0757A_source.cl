#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 288 }
// Out stride: { 18432, 2304, 288, 1 }
// Elementwise input X_T1951 shape: fp32(1, 8, 8, 288):(18432, 2304, 288, 1):72 KiB
// Elementwise input X_T1955 shape: fp32(288):(1):1.125 KiB
// Elementwise input X_I_698 shape: fp32(288):(1):1.125 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1956 = div(X_T1951, X_T1955)
// Elementwise op: [[pid(Add, Switch)]] X_T1957 = add(X_T1956, X_I_698)
// Elementwise op: X_T1958 = cmp_lt(X_T1957, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1959 = cond(X_T1958, X_T2, X_T1957)
// Tile size: { 1, 4, 1, 288 }
// Contraction output var shape: fp32(1, 8, 8, 288):(18432, 2304, 288, 1):72 KiB
// Computed true ops: 73728
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 5120
// Computed mem read: 432
// Computed mem write: 4608
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c51_sdk_638(__global float* restrict  X_T1959, __global const float* restrict  X_T1951, __global const float* restrict  X_T1955, __global const float* restrict  X_I_698)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 4) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((2304 * (i2_gid + i2_tid)) + (288 * i3_gid)) + i4);
      float LX_T1951 = X_T1951[gout_idx];
      float LX_T1955 = X_T1955[i4];
      float LX_I_698 = X_I_698[i4];
      float LX_T1956 = (LX_T1951 / LX_T1955);
      float LX_T1957 = (LX_T1956 + LX_I_698);
      int LX_T1958 = (LX_T1957 < 0.0f);
      float LX_T1959 = select((float)LX_T1957, (float)0.0f, (int)LX_T1958);
      X_T1959[gout_idx] = LX_T1959;
    }
  }
}
