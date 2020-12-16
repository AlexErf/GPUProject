#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 147 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 147, 147, 128 }
// Out stride: { 2765952, 18816, 128, 1 }
// Elementwise input X_T129 shape: fp32(1, 147, 147, 128):(2765952, 18816, 128, 1):10804.5 KiB
// Elementwise input X_T133 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_167 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T134 = div(X_T129, X_T133)
// Elementwise op: [[pid(Add, Switch)]] X_T135 = add(X_T134, X_I_167)
// Elementwise op: X_T136 = cmp_lt(X_T135, X_T2)
// Elementwise op: [[pid(Relu)]] X_T137 = cond(X_T136, X_T2, X_T135)
// Tile size: { 1, 1, 4, 128 }
// Contraction output var shape: fp32(1, 147, 147, 128):(2765952, 18816, 128, 1):10804.5 KiB
// Computed true ops: 11063808
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
__kernel void kernel_c28_sdk_43(__global float* restrict  X_T137, __global const float* restrict  X_T129, __global const float* restrict  X_T133, __global const float* restrict  X_I_167)
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
      float LX_T129 = X_T129[gout_idx];
      float LX_T133 = X_T133[i4];
      float LX_I_167 = X_I_167[i4];
      float LX_T134 = (LX_T129 / LX_T133);
      float LX_T135 = (LX_T134 + LX_I_167);
      int LX_T136 = (LX_T135 < 0.0f);
      float LX_T137 = select((float)LX_T135, (float)0.0f, (int)LX_T136);
      X_T137[gout_idx] = LX_T137;
    }
  }
}
