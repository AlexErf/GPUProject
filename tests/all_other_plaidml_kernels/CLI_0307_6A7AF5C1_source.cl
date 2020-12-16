#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 12 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 14, 14, 384 }
// Out stride: { 75264, 5376, 384, 1 }
// Elementwise input X_T675 shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Elementwise input X_T679 shape: fp32(384):(1):1.5 KiB
// Elementwise input X_I_250 shape: fp32(384):(1):1.5 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T680 = div(X_T675, X_T679)
// Elementwise op: [[pid(Add, Switch)]] X_T681 = add(X_T680, X_I_250)
// Elementwise op: X_T682 = cmp_lt(X_T681, X_T2)
// Elementwise op: [[pid(Relu)]] X_T683 = cond(X_T682, X_T2, X_T681)
// Tile size: { 1, 2, 14, 32 }
// Contraction output var shape: fp32(1, 14, 14, 384):(75264, 5376, 384, 1):294 KiB
// Computed true ops: 301056
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 12, 1
__kernel void kernel_c108_sdk_218(__global float* restrict  X_T683, __global const float* restrict  X_T675, __global const float* restrict  X_T679, __global const float* restrict  X_I_250)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(1) * 32);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i3_lid = 0; i3_lid < 4; i3_lid += 1)
  {
    int i3_cond = ((i3_lid < 3) || (i3_tid < 2));
    if (i3_cond)
    {
      int i3 = ((4 * i3_lid) + i3_tid);
      int gout_idx = (((5376 * (i2_gid + i2_tid)) + (384 * i3)) + (i4_gid + i4_tid));
      float LX_T675 = X_T675[gout_idx];
      float LX_T679 = X_T679[(i4_gid + i4_tid)];
      float LX_I_250 = X_I_250[(i4_gid + i4_tid)];
      float LX_T680 = (LX_T675 / LX_T679);
      float LX_T681 = (LX_T680 + LX_I_250);
      int LX_T682 = (LX_T681 < 0.0f);
      float LX_T683 = select((float)LX_T681, (float)0.0f, (int)LX_T682);
      X_T683[gout_idx] = LX_T683;
    }
  }
}
