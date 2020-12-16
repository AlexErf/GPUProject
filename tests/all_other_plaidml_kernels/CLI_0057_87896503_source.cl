#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 147 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 147, 147, 128 }
// Out stride: { 2765952, 18816, 128, 1 }
// Elementwise input X_T142 shape: fp32(1, 147, 147, 128):(2765952, 18816, 128, 1):10804.5 KiB
// Elementwise input X_T146 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_162 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T147 = div(X_T142, X_T146)
// Elementwise op: [[pid(Add, Switch)]] X_T148 = add(X_T147, X_I_162)
// Tile size: { 1, 1, 4, 128 }
// Contraction output var shape: fp32(1, 147, 147, 128):(2765952, 18816, 128, 1):10804.5 KiB
// Computed true ops: 5531904
// Computed work groups: 5439
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 2048
// Computed mem read: 192
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 147, 1
__kernel void kernel_c28_sdk_47(__global float* restrict  X_T148, __global const float* restrict  X_T142, __global const float* restrict  X_T146, __global const float* restrict  X_I_162)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 4);
  int i3_cond = ((i3_gid != 144) || (i3_tid < 3));
  if (i3_cond)
  {
    for (int i4_lid = 0; i4_lid < 2; i4_lid += 1)
    {
      int i4 = ((64 * i4_lid) + i4_tid);
      int gout_idx = (((18816 * i2_gid) + (128 * (i3_gid + i3_tid))) + i4);
      float LX_T142 = X_T142[gout_idx];
      float LX_T146 = X_T146[i4];
      float LX_I_162 = X_I_162[i4];
      float LX_T147 = (LX_T142 / LX_T146);
      float LX_T148 = (LX_T147 + LX_I_162);
      X_T148[gout_idx] = LX_T148;
    }
  }
}
