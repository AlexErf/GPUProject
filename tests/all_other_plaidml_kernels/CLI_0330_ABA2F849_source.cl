#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 864 }
// Out stride: { 169344, 12096, 864, 1 }
// Elementwise input X_T1003 shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Elementwise input X_T1026 shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Elementwise input X_I_402 shape: fp32(864):(1):3.375 KiB
// Elementwise input X_I_401 shape: fp32(864):(1):3.375 KiB
// Elementwise op: [[pid(Concatenate)]] X_T1027 = add(X_T1003, X_T1026)
// Elementwise op: [[pid(Sub)]] X_T1029 = sub(X_T1027, X_I_402)
// Elementwise op: [[pid(Mul)]] X_T1030 = mul(X_T1029, X_I_401)
// Tile size: { 1, 2, 2, 864 }
// Contraction output var shape: fp32(1, 14, 14, 864):(169344, 12096, 864, 1):661.5 KiB
// Computed true ops: 508032
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1728
// Computed mem write: 27648
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_350(__global float* restrict  X_T1027, __global float* restrict  X_T1030, __global const float* restrict  X_T1003, __global const float* restrict  X_T1026, __global const float* restrict  X_I_402, __global const float* restrict  X_I_401)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 14; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 13) || (i4_tid < 32));
    if (i4_cond)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((12096 * (i2_gid + i2_tid)) + (864 * (i3_gid + i3_tid))) + i4);
      float LX_T1003 = X_T1003[gout_idx];
      float LX_T1026 = X_T1026[gout_idx];
      float LX_I_402 = X_I_402[i4];
      float LX_I_401 = X_I_401[i4];
      float LX_T1027 = (LX_T1003 + LX_T1026);
      float LX_T1029 = (LX_T1027 - LX_I_402);
      float LX_T1030 = (LX_T1029 * LX_I_401);
      X_T1027[gout_idx] = LX_T1027;
      X_T1030[gout_idx] = LX_T1030;
    }
  }
}
