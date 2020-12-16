#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 736 }
// Out stride: { 36064, 5152, 736, 1 }
// Elementwise input X_T1470 shape: fp32(1, 7, 7, 736):(36064, 5152, 736, 1):140.875 KiB
// Elementwise input X_T1474 shape: fp32(736):(1):2.875 KiB
// Elementwise input X_I_561 shape: fp32(736):(1):2.875 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T1475 = div(X_T1470, X_T1474)
// Elementwise op: [[pid(Add, Switch)]] X_T1476 = add(X_T1475, X_I_561)
// Elementwise op: X_T1477 = cmp_lt(X_T1476, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1478 = cond(X_T1477, X_T2, X_T1476)
// Tile size: { 1, 1, 7, 128 }
// Contraction output var shape: fp32(1, 7, 7, 736):(36064, 5152, 736, 1):140.875 KiB
// Computed true ops: 144256
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 336
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c108_sdk_503(__global float* restrict  X_T1478, __global const float* restrict  X_T1470, __global const float* restrict  X_T1474, __global const float* restrict  X_I_561)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 128);
  int i2_gid = get_group_id(1);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 8);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4_cond = ((i4_lid < 3) || (i4_gid != 640));
    if (i4_cond)
    {
      int i4 = ((32 * i4_lid) + i4_tid);
      int i3_cond = (i3_tid < 7);
      if (i3_cond)
      {
        int gout_idx = (((5152 * i2_gid) + (736 * i3_tid)) + (i4_gid + i4));
        float LX_T1470 = X_T1470[gout_idx];
        float LX_T1474 = X_T1474[(i4_gid + i4)];
        float LX_I_561 = X_I_561[(i4_gid + i4)];
        float LX_T1475 = (LX_T1470 / LX_T1474);
        float LX_T1476 = (LX_T1475 + LX_I_561);
        int LX_T1477 = (LX_T1476 < 0.0f);
        float LX_T1478 = select((float)LX_T1476, (float)0.0f, (int)LX_T1477);
        X_T1478[gout_idx] = LX_T1478;
      }
    }
  }
}
