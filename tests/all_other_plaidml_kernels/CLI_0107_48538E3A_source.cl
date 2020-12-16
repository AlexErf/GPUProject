#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 35, 35, 64 }
// Out stride: { 78400, 2240, 64, 1 }
// Elementwise input X_T112 shape: fp32(1, 35, 35, 64):(78400, 2240, 64, 1):306.25 KiB
// Elementwise input X_T116 shape: fp32(64):(1):256 bytes
// Elementwise input X_I_39 shape: fp32(64):(1):256 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T117 = div(X_T112, X_T116)
// Elementwise op: [[pid(Add, Switch)]] X_T118 = add(X_T117, X_I_39)
// Elementwise op: X_T119 = cmp_lt(X_T118, X_T2)
// Elementwise op: [[pid(Relu)]] X_T120 = cond(X_T119, X_T2, X_T118)
// Tile size: { 1, 1, 35, 64 }
// Contraction output var shape: fp32(1, 35, 35, 64):(78400, 2240, 64, 1):306.25 KiB
// Computed true ops: 313600
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 840
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c51_sdk_26(__global float* restrict  X_T120, __global const float* restrict  X_T112, __global const float* restrict  X_T116, __global const float* restrict  X_I_39)
{
  int tid = get_local_id(0);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i3_lid = 0; i3_lid < 5; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 4) || (i3_tid < 3));
    if (i3_cond)
    {
      int i3 = ((8 * i3_lid) + i3_tid);
      for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
      {
        int i4 = ((32 * i4_lid) + i4_tid);
        int gout_idx = (((2240 * i2_gid) + (64 * i3)) + i4);
        float LX_T112 = X_T112[gout_idx];
        float LX_T116 = X_T116[i4];
        float LX_I_39 = X_I_39[i4];
        float LX_T117 = (LX_T112 / LX_T116);
        float LX_T118 = (LX_T117 + LX_I_39);
        int LX_T119 = (LX_T118 < 0.0f);
        float LX_T120 = select((float)LX_T118, (float)0.0f, (int)LX_T119);
        X_T120[gout_idx] = LX_T120;
      }
    }
  }
}
